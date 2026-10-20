#!/usr/bin/env python3
"""Exp134: the Exp132 recipe (GT queries, real member slots, non-member templates) for the v3 head with a pixel branch.

Exp132: Exp131 training plus non-member templates as negatives.

Exp130 showed the closed-loop fill-in mostly revives tracks of false-positive detections: their template is clutter, the clutter is
still there, and a head trained only on member templates answers "present". Here a share of the items uses templates that are
not members: false-positive detections of the training sequences (score >= the floor of the given file, IoU < 0.1 with every GT
box) and, to fill an item, random background boxes of the same size. Their label is always "absent" (no member to find).
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
import exp131_train_head as T131
from exp129_queries import N_FIELDS, NOISE_CAP_SIDES, NOISE_SIDES, SeqGT, draw_age
from models import readout_head as R
from models import readout_head_v2 as V2
from models import readout_head_v3 as V3
import torch.nn.functional as F


def forward_losses_v3(model, head, batch, device, box_weight):
    img_t, img_a, q, lb = [b.to(device) for b in batch]
    b, n = q.shape[:2]
    feat_t, id_t, fine_t = V3.backbone_forward(model, img_t)
    feat_a, id_a, _ = V3.backbone_forward(model, img_a)
    qf, idx = q.reshape(b * n, N_FIELDS), torch.arange(b, device=device).repeat_interleave(n)
    ratio, left, top = lb[idx, 0], lb[idx, 1], lb[idx, 2]
    tb, pred, s0, age, gb, valid = qf[:, 0:4], qf[:, 4:6], qf[:, 6], qf[:, 7], qf[:, 8:12], qf[:, 12]
    inputs = V3.read_inputs(feat_t, id_t, fine_t, img_t, idx, feat_a, id_a, img_a, idx, tb, pred, s0, ratio, left, top, ratio, left, top)
    out = head(*inputs, R.scalars(s0, age))
    gc = (gb[:, :2] + gb[:, 2:]) / 2
    j, i, inside, pos = V3.target_cell(gc, pred, s0)
    inside = inside & (valid > 0)
    ar = torch.arange(b * n, device=device)
    loss_e = F.binary_cross_entropy_with_logits(out['exist'], inside.float())
    loss_c = V3.focal_center_loss(out['center'], j, i, pos, inside)
    k = max(int(inside.sum()), 1)
    off_t = torch.stack([pos[:, 0] - i.float(), pos[:, 1] - j.float()], 1)
    loss_o = ((out['offset'][ar, :, j, i] - off_t).abs().sum(1) * inside).sum() / k / 2
    tgt = torch.stack([gc[:, 0] - gb[:, 0], gc[:, 1] - gb[:, 1], gb[:, 2] - gc[:, 0], gb[:, 3] - gc[:, 1]], 1) / s0[:, None]
    loss_b = ((out['box'][ar, :, j, i] - tgt).abs().sum(1) * inside).sum() / k / 4
    total = loss_e + loss_c + loss_o + box_weight * loss_b
    return total, {'exist': loss_e.item(), 'center': loss_c.item(), 'box': loss_b.item(), 'offset': loss_o.item()}, out['exist'].detach(), inside

CLEAR_SIDES = 1.0   # a background template keeps at least this many of its sides away from every GT centre


class NonMember:
    def __init__(self, fp_file, dataset_dir):
        z = np.load(fp_file)
        self.seq, self.frame, self.box = z['seq'], z['frame'], z['box']
        self.by_frame = {}
        for k, (s, f) in enumerate(zip(self.seq, self.frame)):
            self.by_frame.setdefault((str(s), int(f)), []).append(k)
        self.keys = sorted(self.by_frame)
        self.seqs = {name: SeqGT(Path(dataset_dir) / name) for name in sorted({k[0] for k in self.keys})}

    def background(self, seq, f, s0, rng):
        _, gb = seq.get(f)
        gc = np.c_[(gb[:, 0] + gb[:, 2]) / 2, (gb[:, 1] + gb[:, 3]) / 2] if len(gb) else np.zeros((0, 2))
        c = None
        for _ in range(20):
            c = np.array([rng.uniform(s0, seq.width - s0), rng.uniform(s0, seq.height - s0)])
            if not len(gc) or np.linalg.norm(gc - c, axis=1).min() >= CLEAR_SIDES * s0:
                break
        return np.r_[c - s0 / 2, c + s0 / 2]

    def item(self, rng):
        for _ in range(50):
            name, f_a = self.keys[int(rng.integers(len(self.keys)))]
            seq = self.seqs[name]
            age = draw_age(rng)
            t = f_a + age
            if t <= seq.n_frames:
                break
        idx = self.by_frame[(name, f_a)]
        if len(idx) > T.QUERIES:
            idx = list(rng.choice(idx, T.QUERIES, replace=False))
        boxes = [self.box[k] for k in idx]
        while len(boxes) < T.QUERIES:
            s0 = float(max(boxes[0][2] - boxes[0][0], boxes[0][3] - boxes[0][1]))
            boxes.append(self.background(seq, f_a, s0, rng))
        rows = []
        for b in boxes:
            s0 = float(max(b[2] - b[0], b[3] - b[1]))
            sigma = min(NOISE_SIDES * s0 * np.sqrt(age), NOISE_CAP_SIDES * s0)
            pred = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) + rng.normal(0, sigma, 2)
            rows.append(np.r_[b, pred, s0, age, np.zeros(4), 0.0])
        img_t, lb = T.load_frame(seq, t)
        img_a, _ = T.load_frame(seq, f_a)
        return torch.from_numpy(img_t), torch.from_numpy(img_a), torch.from_numpy(np.asarray(rows, np.float32)), torch.tensor(lb, dtype=torch.float32)


class Mixed3(Dataset):
    def __init__(self, gt_items, real, real_fraction, neg, neg_fraction, n_items, seed):
        self.gt, self.real, self.rf, self.neg, self.nf, self.n, self.seed = gt_items, real, real_fraction, neg, neg_fraction, n_items, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, 11, i])
        u = rng.random()
        if u < self.nf:
            return self.neg.item(rng)
        if u < self.nf + self.rf:
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
    p.add_argument('--real-fraction', type=float, default=0.3)
    p.add_argument('--negatives', type=Path, required=True, help='npz of false-positive detections (seq, frame, box)')
    p.add_argument('--neg-fraction', type=float, default=0.2)
    p.add_argument('--arch', choices=['v1', 'v2', 'v3'], default='v3')
    p.add_argument('--box-weight', type=float, default=T.BOX_WEIGHT)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=T.ITERATIONS)
    p.add_argument('--lr', type=float, default=T.LR)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cuda')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    commit, dirty, source = E.code_identity(Path(__file__).resolve().parents[1])
    torch.manual_seed(T.SEED)
    _, model, _, _ = E.load_model(a.backbone, a.device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    head = {'v1': R.ReadoutHead, 'v2': V2.ReadoutHeadV2, 'v3': V3.ReadoutHeadV3}[a.arch]().to(a.device).train()
    train_seqs = [SeqGT(a.dataset_dir / s) for s in a.train_sequences.split(',')]
    mon_seqs = [SeqGT(a.dataset_dir / s) for s in a.monitor_sequences.split(',')]
    n_items = a.iterations * T.PAIRS
    cv2.setNumThreads(0)
    real = T131.RealSlots(a.real_slots, a.real_states, a.dataset_dir, T.SEED)
    neg = NonMember(a.negatives, a.dataset_dir)
    data = Mixed3(T.PairDataset(train_seqs, n_items, T.SEED), real, a.real_fraction, neg, a.neg_fraction, n_items, T.SEED)
    loader = DataLoader(data, batch_size=T.PAIRS, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0)
    mon_loader = DataLoader(T.PairDataset(mon_seqs, T.MONITOR_BATCHES * T.PAIRS, T.SEED + 1000), batch_size=T.PAIRS, shuffle=False, num_workers=min(2, a.workers))
    if a.arch == 'v3':
        step = lambda batch: forward_losses_v3(model, head, batch, a.device, a.box_weight)
    elif a.arch == 'v2':
        step = lambda batch: T131.forward_losses_v2(model, head, batch, a.device, True, a.box_weight)
    else:
        assert a.box_weight == T.BOX_WEIGHT, 'v1 keeps the Exp129 box weight'
        step = lambda batch: T.forward_losses(model, head, batch, a.device, True)
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
            m = T131.monitor(step, head, mon_loader)
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
        'arch': a.arch, 'box_weight': a.box_weight, 'iterations': a.iterations, 'lr': a.lr, 'seed': T.SEED,
        'real_fraction': a.real_fraction, 'real_slots_sha256': hashlib.sha256(a.real_slots.read_bytes()).hexdigest(),
        'neg_fraction': a.neg_fraction, 'negatives': str(a.negatives), 'negatives_sha256': hashlib.sha256(a.negatives.read_bytes()).hexdigest(),
        'negative_frames': len(neg.keys), 'backbone_sha256': hashlib.sha256(a.backbone.read_bytes()).hexdigest(),
        'train_sequences': a.train_sequences.split(','), 'code_commit': commit, 'tracked_dirty': dirty, 'code_source': source,
        'torch': torch.__version__, 'device': a.device, 'final_monitor': m}, indent=2) + '\n')
    tb.close()


if __name__ == '__main__':
    main()
