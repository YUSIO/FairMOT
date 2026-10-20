#!/usr/bin/env python3
"""Exp131: train the read-out head on GT-built slot queries (Exp129) mixed with real tracker slots from training sequences.

Real slots come from experiments/exp131_readout_real_states/data_prep (ByteTrack t0.5 on the FairMOT cache of the training
sequences, Exp128 states.py, Exp129 make_slots.py). A real item is one (sequence, t, template frame) group of slots; groups
with fewer than QUERIES slots are padded with GT-built queries of the same frame pair. Labels are geometric, as in Exp129.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import exp128_dense_export as E
import exp129_train_head as T
from exp129_queries import N_FIELDS, SeqGT, sample_queries
from models import readout_head as R
from models import readout_head_v2 as V2
import torch.nn.functional as F


def forward_losses_v2(model, head, batch, device, use_template, box_weight):
    img_t, img_a, q, lb = [b.to(device) for b in batch]
    b, n = q.shape[:2]
    feat_t, id_t, fine_t = V2.backbone_forward(model, img_t)
    feat_a, id_a, _ = V2.backbone_forward(model, img_a)
    qf, idx = q.reshape(b * n, N_FIELDS), torch.arange(b, device=device).repeat_interleave(n)
    ratio, left, top = lb[idx, 0], lb[idx, 1], lb[idx, 2]
    tb, pred, s0, age, gb, valid = qf[:, 0:4], qf[:, 4:6], qf[:, 6], qf[:, 7], qf[:, 8:12], qf[:, 12]
    inputs = V2.read_inputs(feat_t, id_t, fine_t, idx, feat_a, id_a, idx, tb, pred, s0, ratio, left, top, ratio, left, top)
    out = head(*inputs, R.scalars(s0, age), use_template)
    gc = (gb[:, :2] + gb[:, 2:]) / 2
    j, i, inside, pos = V2.target_cell(gc, pred, s0)
    inside = inside & (valid > 0)
    ar = torch.arange(b * n, device=device)
    loss_e = F.binary_cross_entropy_with_logits(out['exist'], inside.float())
    loss_c = V2.focal_center_loss(out['center'], j, i, pos, inside)
    k = max(int(inside.sum()), 1)
    off_t = torch.stack([pos[:, 0] - i.float(), pos[:, 1] - j.float()], 1)
    loss_o = ((out['offset'][ar, :, j, i] - off_t).abs().sum(1) * inside).sum() / k / 2
    tgt = torch.stack([gc[:, 0] - gb[:, 0], gc[:, 1] - gb[:, 1], gb[:, 2] - gc[:, 0], gb[:, 3] - gc[:, 1]], 1) / s0[:, None]
    loss_b = ((out['box'][ar, :, j, i] - tgt).abs().sum(1) * inside).sum() / k / 4
    total = loss_e + loss_c + loss_o + box_weight * loss_b
    return total, {'exist': loss_e.item(), 'center': loss_c.item(), 'box': loss_b.item(), 'offset': loss_o.item()}, out['exist'].detach(), inside


@torch.no_grad()
def monitor(fn, head, loader):
    from scipy.stats import rankdata
    head.eval()
    tot, logits, labels = {}, [], []
    for k, batch in enumerate(loader):
        if k >= T.MONITOR_BATCHES:
            break
        _, parts, lg, lab = fn(batch)
        for key, v in parts.items():
            tot[key] = tot.get(key, 0.) + v
        logits.append(lg.cpu()); labels.append(lab.cpu())
    head.train()
    s, y = torch.cat(logits).numpy(), torch.cat(labels).numpy().astype(int)
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    auc = float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')
    return {**{k: v / T.MONITOR_BATCHES for k, v in tot.items()}, 'auc_exist': auc}


class RealSlots(Dataset):
    def __init__(self, slots, states, dataset_dir, seed):
        z = np.load(slots)
        st = json.loads(Path(states).read_text())
        assert len(st) == len(z['t'])
        self.rows, groups = [], {}
        for i, s in enumerate(st):
            g = s['gt_box']
            self.rows.append(np.r_[z['tbox'][i], z['pred'][i], z['s0'][i], z['age'][i], g if g else np.zeros(4), 1.0 if g else 0.0].astype(np.float32))
            groups.setdefault((s['seq'], int(z['t'][i]), int(z['f0'][i])), []).append(i)
        self.keys = sorted(groups)
        self.groups = groups
        self.seqs = {name: SeqGT(Path(dataset_dir) / name) for name in sorted({k[0] for k in self.keys})}
        self.seed = seed

    def item(self, rng):
        seq_name, t, f0 = self.keys[int(rng.integers(len(self.keys)))]
        idx = self.groups[(seq_name, t, f0)]
        if len(idx) > T.QUERIES:
            idx = list(rng.choice(idx, T.QUERIES, replace=False))
        q = np.stack([self.rows[i] for i in idx])
        if len(q) < T.QUERIES:
            pad = sample_queries(self.seqs[seq_name], t, t - f0, rng, T.QUERIES - len(q))
            q = np.concatenate([q, pad]) if pad is not None else np.concatenate([q, q[rng.integers(0, len(q), T.QUERIES - len(q))]])
        seq = self.seqs[seq_name]
        img_t, lb = T.load_frame(seq, t)
        img_a, _ = T.load_frame(seq, f0)
        return torch.from_numpy(img_t), torch.from_numpy(img_a), torch.from_numpy(q.astype(np.float32)), torch.tensor(lb, dtype=torch.float32)


class Mixed(Dataset):
    def __init__(self, gt_items, real, real_fraction, n_items, seed):
        self.gt, self.real, self.frac, self.n, self.seed = gt_items, real, real_fraction, n_items, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, 7, i])
        if rng.random() < self.frac:
            return self.real.item(rng)
        return self.gt[i]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--train-sequences', required=True)
    p.add_argument('--monitor-sequences', required=True)
    p.add_argument('--real-slots', type=Path, required=True)
    p.add_argument('--real-states', type=Path, required=True)
    p.add_argument('--real-fraction', type=float, default=0.5)
    p.add_argument('--init', type=Path, help='start from these head weights instead of a fresh head')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--no-template', action='store_true')
    p.add_argument('--iterations', type=int, default=T.ITERATIONS)
    p.add_argument('--lr', type=float, default=T.LR)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cuda')
    p.add_argument('--arch', choices=['v1', 'v2'], default='v1')
    p.add_argument('--box-weight', type=float, default=T.BOX_WEIGHT)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    commit, dirty, source = E.code_identity(Path(__file__).resolve().parents[1])
    torch.manual_seed(T.SEED)
    _, model, _, _ = E.load_model(a.backbone, a.device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    head = (V2.ReadoutHeadV2() if a.arch == 'v2' else R.ReadoutHead()).to(a.device)
    if a.init:
        head.load_state_dict(torch.load(a.init, map_location=a.device))
    head.train()
    train_seqs = [SeqGT(a.dataset_dir / s) for s in a.train_sequences.split(',')]
    mon_seqs = [SeqGT(a.dataset_dir / s) for s in a.monitor_sequences.split(',')]
    n_items = a.iterations * T.PAIRS
    cv2.setNumThreads(0)
    real = RealSlots(a.real_slots, a.real_states, a.dataset_dir, T.SEED)
    data = Mixed(T.PairDataset(train_seqs, n_items, T.SEED), real, a.real_fraction, n_items, T.SEED)
    loader = DataLoader(data, batch_size=T.PAIRS, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0)
    mon_loader = DataLoader(T.PairDataset(mon_seqs, T.MONITOR_BATCHES * T.PAIRS, T.SEED + 1000), batch_size=T.PAIRS, shuffle=False, num_workers=min(2, a.workers))
    if a.arch == 'v2':
        step = lambda batch: forward_losses_v2(model, head, batch, a.device, not a.no_template, a.box_weight)
    else:
        assert a.box_weight == T.BOX_WEIGHT, 'v1 keeps the Exp129 box weight'
        step = lambda batch: T.forward_losses(model, head, batch, a.device, not a.no_template)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=T.WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iterations)
    tb = SummaryWriter(str(a.out / 'tb'))
    log = (a.out / 'train_log.jsonl').open('w')
    start, m = time.monotonic(), None
    for it, batch in enumerate(loader, 1):
        loss, parts, _, _ = step(batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        rec = {'iter': it, 'loss': loss.item(), **parts, 'lr': sched.get_last_lr()[0], 'elapsed_s': round(time.monotonic() - start, 1)}
        log.write(json.dumps(rec) + '\n')
        for k in [x for x in rec if x not in ('iter', 'elapsed_s')]:
            tb.add_scalar(f'train/{k}', rec[k], it)
        if it % T.MONITOR_EVERY == 0 or it == a.iterations:
            m = monitor(step, head, mon_loader)
            for k, v in m.items():
                tb.add_scalar(f'monitor/{k}', v, it)
            log.write(json.dumps({'iter': it, 'monitor': m}) + '\n')
            log.flush()
            print(f'iter {it}: loss {rec["loss"]:.4f} monitor {m}', flush=True)
        if it == a.iterations:
            break
    torch.save(head.state_dict(), a.out / 'head_final.pt')
    (a.out / 'train_manifest.json').write_text(json.dumps({
        'head_parameters': R.n_parameters(head), 'head_sha256': hashlib.sha256((a.out / 'head_final.pt').read_bytes()).hexdigest(),
        'iterations': a.iterations, 'lr': a.lr, 'arch': a.arch, 'box_weight': a.box_weight, 'seed': T.SEED, 'use_template': not a.no_template, 'init': str(a.init) if a.init else None,
        'real_fraction': a.real_fraction, 'real_slots': str(a.real_slots), 'real_slots_sha256': hashlib.sha256(a.real_slots.read_bytes()).hexdigest(),
        'real_groups': len(real.keys), 'backbone': str(a.backbone), 'backbone_sha256': hashlib.sha256(a.backbone.read_bytes()).hexdigest(),
        'train_sequences': a.train_sequences.split(','), 'monitor_sequences': a.monitor_sequences.split(','),
        'code_commit': commit, 'tracked_dirty': dirty, 'code_source': source, 'torch': torch.__version__, 'device': a.device,
        'final_monitor': m}, indent=2) + '\n')
    tb.close()


if __name__ == '__main__':
    main()
