#!/usr/bin/env python3
"""Exp135: the Exp134 recipe (v3 head; GT queries, real member slots, non-member templates) plus hard real slots from degraded copies
of the training sequences.

Exp135's probe: on an FN-heavy cell (mildly degraded fit sequences) the Exp134 head stays below the crop verifier at every
threshold, even at p 0.05 (+1.01 MOTA against +1.28) and even though it was trained on the clean versions of those sequences.
Its training slots come from sequences the detector has seen, where members it loses are few (659 present states on 30
sequences). Degraded copies (down/up-scaling, blur, contrast, brightness, JPEG; per-sequence parameters, Exp135 code/degrade.py)
make the frozen detector lose members it would find on the clean frames: the tracker's lost slots on them are the hard queries
the closed loop asks (present 3,364 mild / 4,920 strong on 30 sequences). Each hard source is (slots, states, dataset dir of the
degraded images); its labels are geometric like every other real slot.

All sources are restricted to --train-sequences, so a development head can leave sequences out.
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
import exp132_train_head as T132
import exp134_train_head as T134
from exp129_queries import SeqGT
from models import readout_head as R
from models import readout_head_v3 as V3


def restrict(source, seqs):
    """Keep the items of a RealSlots / NonMember source whose sequence is in seqs (keys are (seq, ...) tuples)."""
    source.keys = [k for k in source.keys if str(k[0]) in seqs]
    assert source.keys, 'no items left after restricting to the training sequences'
    return source


class Mixed5(Dataset):
    """GT queries, clean real slots, hard (degraded) real slots, non-member templates. A hard item picks a source in proportion to
    its number of slot groups."""

    def __init__(self, gt_items, real, real_fraction, hard, hard_fraction, neg, neg_fraction, n_items, seed):
        self.gt, self.real, self.rf, self.hard, self.hf, self.neg, self.nf = gt_items, real, real_fraction, hard, hard_fraction, neg, neg_fraction
        self.w = np.array([len(h.keys) for h in hard], float)
        self.w /= self.w.sum()
        self.n, self.seed = n_items, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, 13, i])
        u = rng.random()
        if u < self.nf:
            return self.neg.item(rng)
        if u < self.nf + self.rf:
            return self.real.item(rng)
        if u < self.nf + self.rf + self.hf:
            return self.hard[int(rng.choice(len(self.hard), p=self.w))].item(rng)
        return self.gt[i]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--train-sequences', required=True)
    p.add_argument('--monitor-sequences', required=True)
    p.add_argument('--real-slots', type=Path, required=True)
    p.add_argument('--real-states', type=Path, required=True)
    p.add_argument('--real-fraction', type=float, default=0.15)
    p.add_argument('--hard', action='append', required=True, help='slots.npz:states.json:dataset_dir of degraded images (repeatable)')
    p.add_argument('--hard-fraction', type=float, default=0.25)
    p.add_argument('--negatives', type=Path, required=True, help='npz of false-positive detections (seq, frame, box)')
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
    _, model, _, _ = E.load_model(a.backbone, a.device)
    for prm in model.parameters():
        prm.requires_grad_(False)
    head = V3.ReadoutHeadV3().to(a.device).train()
    seqs = a.train_sequences.split(',')
    train_seqs = [SeqGT(a.dataset_dir / s) for s in seqs]
    mon_seqs = [SeqGT(a.dataset_dir / s) for s in a.monitor_sequences.split(',')]
    n_items = a.iterations * T.PAIRS
    cv2.setNumThreads(0)
    real = restrict(T131.RealSlots(a.real_slots, a.real_states, a.dataset_dir, T.SEED), set(seqs))
    hard_specs = [h.split(':') for h in a.hard]
    hard = [restrict(T131.RealSlots(Path(s), Path(st), Path(d), T.SEED), set(seqs)) for s, st, d in hard_specs]
    neg = restrict(T132.NonMember(a.negatives, a.dataset_dir), set(seqs))
    data = Mixed5(T.PairDataset(train_seqs, n_items, T.SEED), real, a.real_fraction, hard, a.hard_fraction, neg, a.neg_fraction, n_items, T.SEED)
    loader = DataLoader(data, batch_size=T.PAIRS, shuffle=False, num_workers=a.workers, persistent_workers=a.workers > 0)
    mon_loader = DataLoader(T.PairDataset(mon_seqs, T.MONITOR_BATCHES * T.PAIRS, T.SEED + 1000), batch_size=T.PAIRS, shuffle=False, num_workers=min(2, a.workers))
    step = lambda batch: T134.forward_losses_v3(model, head, batch, a.device, a.box_weight)
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
        'head_parameters': R.n_parameters(head), 'head_sha256': sha(a.out / 'head_final.pt'),
        'arch': 'v3', 'box_weight': a.box_weight, 'iterations': a.iterations, 'lr': a.lr, 'seed': T.SEED,
        'real_fraction': a.real_fraction, 'real_slots_sha256': sha(a.real_slots), 'real_groups': len(real.keys),
        'hard_fraction': a.hard_fraction,
        'hard': [{'slots': s, 'slots_sha256': sha(s), 'states': st, 'dataset_dir': d, 'groups': len(h.keys)} for (s, st, d), h in zip(hard_specs, hard)],
        'neg_fraction': a.neg_fraction, 'negatives': str(a.negatives), 'negatives_sha256': sha(a.negatives), 'negative_frames': len(neg.keys),
        'backbone_sha256': sha(a.backbone), 'train_sequences': seqs, 'code_commit': commit, 'tracked_dirty': dirty, 'code_source': source,
        'torch': torch.__version__, 'device': a.device, 'final_monitor': m}, indent=2) + '\n')
    tb.close()


if __name__ == '__main__':
    main()
