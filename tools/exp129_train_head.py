#!/usr/bin/env python3
"""Exp129: train the read-out head on a frozen FairMOT backbone with queries built from ground-truth trajectories.

The backbone, Re-ID head and detection heads are frozen and in eval mode; only ReadoutHead is trained. The last
iteration's weights are saved (no checkpoint selection). TensorBoard events go to <out>/tb.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import exp128_dense_export as E                      # sets up src/lib on sys.path and loads the frozen model
from datasets.dataset.jde import letterbox
from exp129_queries import MIX_ABSENT, N_FIELDS, P_ABSENT_ITEM, SeqGT, absent_item, draw_age, sample_queries
from models import readout_head as R

SEED = 129
ITERATIONS = 8000
PAIRS = 8
QUERIES = 8
LR, WD = 1e-3, 1e-4
BOX_WEIGHT = 0.1
MONITOR_EVERY = 500
MONITOR_BATCHES = 25


def load_frame(seq, f):
    img0 = cv2.imread(str(seq.dir / 'img1' / f'{f:06d}.jpg'))
    assert img0 is not None, f'missing frame {f} of {seq.dir.name}'
    img, ratio, dw, dh = letterbox(img0, height=R.IMG_H, width=R.IMG_W)
    img = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    return img, R.letterbox_params(*img0.shape[:2])


class PairDataset(Dataset):
    def __init__(self, seqs, n_items, seed):
        self.seqs, self.n, self.seed = seqs, n_items, seed
        w = np.array([s.n_frames for s in seqs], float)
        self.w = w / w.sum()

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, i])
        for _ in range(100):
            seq = self.seqs[int(rng.choice(len(self.seqs), p=self.w))]
            item = absent_item(seq, rng) if rng.random() < P_ABSENT_ITEM else None
            if item is not None:
                t, age = item
                q = sample_queries(seq, t, age, rng, QUERIES, MIX_ABSENT)
            else:
                t = int(rng.integers(2, seq.n_frames + 1))
                age = min(draw_age(rng), t - 1)
                q = sample_queries(seq, t, age, rng, QUERIES)
            if q is not None:
                break
        img_t, lb = load_frame(seq, t)
        img_a, _ = load_frame(seq, t - age)
        return torch.from_numpy(img_t), torch.from_numpy(img_a), torch.from_numpy(q), torch.tensor(lb, dtype=torch.float32)


def forward_losses(model, head, batch, device, use_template):
    img_t, img_a, q, lb = [b.to(device) for b in batch]
    b, n = q.shape[:2]
    feat_t, id_t = R.backbone_forward(model, img_t)
    feat_a, id_a = R.backbone_forward(model, img_a)
    qf, idx = q.reshape(b * n, N_FIELDS), torch.arange(b, device=device).repeat_interleave(n)
    ratio, left, top = lb[idx, 0], lb[idx, 1], lb[idx, 2]
    tb, pred, s0, age, gb, valid = qf[:, 0:4], qf[:, 4:6], qf[:, 6], qf[:, 7], qf[:, 8:12], qf[:, 12]
    inputs = R.read_inputs(feat_t, id_t, idx, feat_a, id_a, idx, tb, pred, s0, ratio, left, top, ratio, left, top)
    out = head(*inputs, R.scalars(s0, age), use_template)
    gc = (gb[:, :2] + gb[:, 2:]) / 2
    j, i, inside, pos = R.target_cell(gc, pred, s0)
    inside = inside & (valid > 0)
    loss_e = F.binary_cross_entropy_with_logits(out['exist'], inside.float())
    loss_c = R.focal_center_loss(out['center'], j, i, pos, inside)
    cell = pred + (torch.stack([i, j], 1).float() + .5) / R.WIN * R.WINDOW_SIDES * s0[:, None] - R.WINDOW_SIDES / 2 * s0[:, None]
    tgt = torch.stack([cell[:, 0] - gb[:, 0], cell[:, 1] - gb[:, 1], gb[:, 2] - cell[:, 0], gb[:, 3] - cell[:, 1]], 1) / s0[:, None]
    got = out['box'][torch.arange(b * n, device=device), :, j, i]
    loss_b = ((got - tgt).abs().sum(1) * inside).sum() / max(int(inside.sum()), 1) / 4
    return loss_e + loss_c + BOX_WEIGHT * loss_b, {'exist': loss_e.item(), 'center': loss_c.item(), 'box': loss_b.item()}, out['exist'].detach(), inside


@torch.no_grad()
def monitor(model, head, loader, device, use_template):
    head.eval()
    tot, logits, labels = {'exist': 0., 'center': 0., 'box': 0.}, [], []
    for k, batch in enumerate(loader):
        if k >= MONITOR_BATCHES:
            break
        _, parts, lg, lab = forward_losses(model, head, batch, device, use_template)
        for key in tot:
            tot[key] += parts[key]
        logits.append(lg.cpu()); labels.append(lab.cpu())
    head.train()
    from scipy.stats import rankdata
    s, y = torch.cat(logits).numpy(), torch.cat(labels).numpy().astype(int)
    n1, n0 = int(y.sum()), len(y) - int(y.sum())
    auc = float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float('nan')
    return {**{k: v / MONITOR_BATCHES for k, v in tot.items()}, 'auc_exist': auc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--dataset-dir', type=Path, required=True, help='directory with UAVSwarm-NN sequence folders')
    p.add_argument('--train-sequences', required=True)
    p.add_argument('--monitor-sequences', required=True, help='held-out sequences, GT-built queries, monitoring only')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--no-template', action='store_true')
    p.add_argument('--iterations', type=int, default=ITERATIONS)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cuda')
    p.add_argument('--monitor-dataset-dir', type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip())
    torch.manual_seed(SEED)
    _, model, state, _ = E.load_model(a.backbone, a.device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    head = R.ReadoutHead().to(a.device).train()
    n_params = R.n_parameters(head)
    assert n_params <= 1_000_000, n_params
    train_seqs = [SeqGT(a.dataset_dir / s) for s in a.train_sequences.split(',')]
    mon_seqs = [SeqGT((a.monitor_dataset_dir or a.dataset_dir) / s) for s in a.monitor_sequences.split(',')]
    n_items = a.iterations * PAIRS
    cv2.setNumThreads(0)
    loader = DataLoader(PairDataset(train_seqs, n_items, SEED), batch_size=PAIRS, shuffle=False, num_workers=a.workers,
                        persistent_workers=a.workers > 0)
    mon_loader = DataLoader(PairDataset(mon_seqs, MONITOR_BATCHES * PAIRS, SEED + 1000), batch_size=PAIRS, shuffle=False,
                            num_workers=min(2, a.workers))
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iterations)
    tb = SummaryWriter(str(a.out / 'tb'))
    log = (a.out / 'train_log.jsonl').open('w')
    start = time.monotonic()
    for it, batch in enumerate(loader, 1):
        loss, parts, _, _ = forward_losses(model, head, batch, a.device, not a.no_template)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        rec = {'iter': it, 'loss': loss.item(), **parts, 'lr': sched.get_last_lr()[0], 'elapsed_s': round(time.monotonic() - start, 1)}
        log.write(json.dumps(rec) + '\n')
        for k in ('loss', 'exist', 'center', 'box', 'lr'):
            tb.add_scalar(f'train/{k}', rec[k], it)
        if it % MONITOR_EVERY == 0 or it == a.iterations:
            m = monitor(model, head, mon_loader, a.device, not a.no_template)
            for k, v in m.items():
                tb.add_scalar(f'monitor/{k}', v, it)
            log.write(json.dumps({'iter': it, 'monitor': m}) + '\n')
            log.flush()
            print(f'iter {it}: loss {rec["loss"]:.4f} monitor {m}', flush=True)
        if it == a.iterations:
            break
    torch.save(head.state_dict(), a.out / 'head_final.pt')
    digest = hashlib.sha256((a.out / 'head_final.pt').read_bytes()).hexdigest()
    (a.out / 'train_manifest.json').write_text(json.dumps({
        'head_parameters': n_params, 'head_sha256': digest, 'iterations': a.iterations, 'seed': SEED, 'use_template': not a.no_template,
        'backbone': str(a.backbone), 'backbone_sha256': hashlib.sha256(a.backbone.read_bytes()).hexdigest(),
        'train_sequences': a.train_sequences.split(','), 'monitor_sequences': a.monitor_sequences.split(','),
        'code_commit': commit, 'tracked_dirty': dirty, 'torch': torch.__version__, 'device': a.device,
        'final_monitor': m if a.iterations else None}, indent=2) + '\n')
    tb.close()


if __name__ == '__main__':
    main()
