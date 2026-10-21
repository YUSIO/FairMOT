#!/usr/bin/env python3
"""Exp137: the Exp134 recipe (v3 head; GT queries 0.5, real member slots 0.3, non-member templates 0.2) on a frozen YOLOX detector.

Datasets are the Exp129 / Exp131 / Exp132 classes; their frame loader (exp129_train_head.load_frame) is replaced by the YOLOX one,
and the backbone forward and the window / template sampling come from models/readout_yolox.py. Every source is restricted to
--train-sequences.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import exp128_dense_export as E
import exp129_train_head as T
import exp131_train_head as T131
import exp132_train_head as T132
from exp129_queries import N_FIELDS, SeqGT
from exp135_train_head import restrict
from models import readout_head as R
from models import readout_yolox as Y

T.load_frame = Y.load_frame  # every dataset class reads frames through this module attribute


def forward_losses(model, head, batch, device, box_weight):
    img_t, img_a, q, lb = [b.to(device) for b in batch]
    b, n = q.shape[:2]
    feat_t, id_t, fine_t = Y.backbone_forward(model, img_t)
    feat_a, id_a, _ = Y.backbone_forward(model, img_a)
    qf, idx = q.reshape(b * n, N_FIELDS), torch.arange(b, device=device).repeat_interleave(n)
    ratio, left, top = lb[idx, 0], lb[idx, 1], lb[idx, 2]
    tb, pred, s0, age, gb, valid = qf[:, 0:4], qf[:, 4:6], qf[:, 6], qf[:, 7], qf[:, 8:12], qf[:, 12]
    inputs = Y.read_inputs(feat_t, id_t, fine_t, img_t, idx, feat_a, id_a, img_a, idx, tb, pred, s0, ratio, left, top, ratio, left, top)
    out = head(*inputs, R.scalars(s0, age))
    gc = (gb[:, :2] + gb[:, 2:]) / 2
    j, i, inside, pos = Y.target_cell(gc, pred, s0)
    inside = inside & (valid > 0)
    ar = torch.arange(b * n, device=device)
    loss_e = F.binary_cross_entropy_with_logits(out['exist'], inside.float())
    loss_c = Y.focal_center_loss(out['center'], j, i, pos, inside)
    k = max(int(inside.sum()), 1)
    off_t = torch.stack([pos[:, 0] - i.float(), pos[:, 1] - j.float()], 1)
    loss_o = ((out['offset'][ar, :, j, i] - off_t).abs().sum(1) * inside).sum() / k / 2
    tgt = torch.stack([gc[:, 0] - gb[:, 0], gc[:, 1] - gb[:, 1], gb[:, 2] - gc[:, 0], gb[:, 3] - gc[:, 1]], 1) / s0[:, None]
    loss_b = ((out['box'][ar, :, j, i] - tgt).abs().sum(1) * inside).sum() / k / 4
    total = loss_e + loss_c + loss_o + box_weight * loss_b
    return total, {'exist': loss_e.item(), 'center': loss_c.item(), 'box': loss_b.item(), 'offset': loss_o.item()}, out['exist'].detach(), inside


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bytetrack-root', type=Path, required=True)
    p.add_argument('--exp-file', required=True, help='YOLOX exp file, relative to --bytetrack-root')
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--train-sequences', required=True)
    p.add_argument('--monitor-dir', type=Path, required=True)
    p.add_argument('--monitor-sequences', required=True)
    p.add_argument('--real-slots', type=Path, required=True)
    p.add_argument('--real-states', type=Path, required=True)
    p.add_argument('--real-fraction', type=float, default=0.3)
    p.add_argument('--negatives', type=Path, required=True)
    p.add_argument('--neg-fraction', type=float, default=0.2)
    p.add_argument('--box-weight', type=float, default=1.0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=T.ITERATIONS)
    p.add_argument('--lr', type=float, default=T.LR)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cuda')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    commit, dirty, source = E.code_identity(Path(__file__).resolve().parents[1])
    torch.manual_seed(T.SEED)
    model = Y.load_model(a.bytetrack_root, a.exp_file, a.ckpt, a.device)
    head = Y.ReadoutHeadV3().to(a.device).train()
    seqs = a.train_sequences.split(',')
    train_seqs = [SeqGT(a.dataset_dir / s) for s in seqs]
    mon_seqs = [SeqGT(a.monitor_dir / s) for s in a.monitor_sequences.split(',')]
    n_items = a.iterations * T.PAIRS
    cv2.setNumThreads(0)
    real = restrict(T131.RealSlots(a.real_slots, a.real_states, a.dataset_dir, T.SEED), set(seqs))
    neg = restrict(T132.NonMember(a.negatives, a.dataset_dir), set(seqs))
    data = T132.Mixed3(T.PairDataset(train_seqs, n_items, T.SEED), real, a.real_fraction, neg, a.neg_fraction, n_items, T.SEED)
    loader = DataLoader(data, batch_size=T.PAIRS, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0)
    mon_loader = DataLoader(T.PairDataset(mon_seqs, T.MONITOR_BATCHES * T.PAIRS, T.SEED + 1000), batch_size=T.PAIRS, shuffle=False, num_workers=min(2, a.workers))
    step = lambda batch: forward_losses(model, head, batch, a.device, a.box_weight)
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
        'head_parameters': R.n_parameters(head), 'head_sha256': sha(a.out / 'head_final.pt'), 'arch': 'v3y', 'detector': 'yolox',
        'exp_file': a.exp_file, 'ckpt_sha256': sha(a.ckpt), 'box_weight': a.box_weight, 'iterations': a.iterations, 'lr': a.lr,
        'seed': T.SEED, 'real_fraction': a.real_fraction, 'real_slots_sha256': sha(a.real_slots), 'real_groups': len(real.keys),
        'neg_fraction': a.neg_fraction, 'negatives': str(a.negatives), 'negatives_sha256': sha(a.negatives), 'negative_frames': len(neg.keys),
        'train_sequences': seqs, 'code_commit': commit, 'tracked_dirty': dirty, 'code_source': source,
        'torch': torch.__version__, 'device': a.device, 'final_monitor': m}, indent=2) + '\n')
    tb.close()


if __name__ == '__main__':
    main()
