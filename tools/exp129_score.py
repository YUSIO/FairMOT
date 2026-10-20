#!/usr/bin/env python3
"""Exp129: score slots with the read-out head arms and the detector window arm.

Input: slots.npz written by experiments/exp129_unified_readout_head_probe/code/make_slots.py (seq, t, f0, tbox, pred, s0,
age, kind). Output: scores.npz with <arm>_p (n,) and <arm>_box (n, 4, x1 y1 x2 y2, NaN when no box) aligned to the slots.

Arms: head (own template), head_shuffled (template of another slot of the same sequence, seed 129), head_notempl (a
separately trained head, template zeroed), det_win (highest detector peak, from the Exp128 dense export, inside the same window).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv2
import numpy as np
import torch

import exp128_dense_export as E
from exp129_train_head import load_frame
from exp129_queries import SeqGT
from models import readout_head as R

SEED = 129
CHUNK = 32


class Bank:
    """Backbone outputs of the frames a chunk needs, with per-frame letterbox parameters."""

    def __init__(self, model, seq, frames, device):
        self.index = {f: k for k, f in enumerate(sorted(frames))}
        imgs, params = [], []
        for f in sorted(frames):
            img, lb = load_frame(seq, f)
            imgs.append(torch.from_numpy(img)); params.append(lb)
        feats, ids = [], []
        for k in range(0, len(imgs), 4):
            f_, i_ = R.backbone_forward(model, torch.stack(imgs[k:k + 4]).to(device))
            feats.append(f_); ids.append(i_)
        self.feat, self.id = torch.cat(feats), torch.cat(ids)
        self.params = torch.tensor(params, dtype=torch.float32, device=device)

    def idx(self, frames, device):
        return torch.tensor([self.index[int(f)] for f in frames], device=device)


def load_head(path, device):
    head = R.ReadoutHead().to(device)
    head.load_state_dict(torch.load(path, map_location=device))
    return head.eval()


def det_window(dense, t, pred, s0):
    a, b = dense['starts'][t - 1], dense['starts'][t]
    dets = dense['dets'][a:b]
    if len(dets) == 0:
        return 0.0, np.full(4, np.nan)
    c = np.c_[(dets[:, 0] + dets[:, 2]) / 2, (dets[:, 1] + dets[:, 3]) / 2]
    inside = np.abs(c - pred).max(1) <= R.WINDOW_SIDES / 2 * s0
    if not inside.any():
        return 0.0, np.full(4, np.nan)
    k = np.where(inside)[0][dets[inside, 4].argmax()]
    return float(dets[k, 4]), dets[k, :4].astype(float)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--slots', type=Path, required=True)
    p.add_argument('--dataset-dir', type=Path, required=True, help='directory with the UAVSwarm-NN folders of this cell')
    p.add_argument('--backbone', type=Path, required=True)
    p.add_argument('--head', action='append', default=[], help='name=path for head and/or head_notempl')
    p.add_argument('--dense', type=Path, help='Exp128 dense export of this cell (for det_win)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cuda')
    a = p.parse_args()
    z = np.load(a.slots, allow_pickle=False)
    seqs_all, t_all, f0_all = z['seq'], z['t'], z['f0']
    tbox_all, pred_all, s0_all, age_all = z['tbox'], z['pred'], z['s0'], z['age']
    n = len(seqs_all)
    _, model, _, _ = E.load_model(a.backbone, a.device)
    heads = {}
    for spec in a.head:
        name, path = spec.split('=')
        heads[name] = load_head(Path(path), a.device)
    res = {}
    arms = [k for k in ('head', 'head_shuffled', 'head_notempl') if (k if k != 'head_shuffled' else 'head') in heads]
    for arm in arms + (['det_win'] if a.dense else []):
        res[arm] = (np.full(n, np.nan), np.full((n, 4), np.nan))
    rng = np.random.default_rng(SEED)
    for seq_name in sorted(set(seqs_all)):
        rows = np.where(seqs_all == seq_name)[0]
        seq = SeqGT(a.dataset_dir / seq_name)
        # shuffled template: the template of another slot of the same sequence (a cyclic shift is a derangement)
        other = np.roll(rows, int(rng.integers(1, len(rows)))) if len(rows) > 1 else rows
        dense = None
        if a.dense:
            with np.load(a.dense / (seq_name + '.npz')) as d:
                dense = {k: d[k] for k in ('dets', 'starts')}
            for r in rows:
                pr, bx = det_window(dense, int(t_all[r]), pred_all[r], s0_all[r])
                res['det_win'][0][r], res['det_win'][1][r] = pr, bx
        if not heads:
            continue
        for k in range(0, len(rows), CHUNK):
            sel, oth = rows[k:k + CHUNK], other[k:k + CHUNK]
            frames = set(t_all[sel].tolist()) | set(f0_all[sel].tolist()) | set(f0_all[oth].tolist())
            bank = Bank(model, seq, frames, a.device)
            dev = a.device
            idx_t, idx_a, idx_o = bank.idx(t_all[sel], dev), bank.idx(f0_all[sel], dev), bank.idx(f0_all[oth], dev)
            pred = torch.tensor(pred_all[sel], dtype=torch.float32, device=dev)
            s0 = torch.tensor(s0_all[sel], dtype=torch.float32, device=dev)
            age = torch.tensor(age_all[sel], dtype=torch.float32, device=dev)
            lt, la, lo = bank.params[idx_t], bank.params[idx_a], bank.params[idx_o]
            tb = torch.tensor(tbox_all[sel], dtype=torch.float32, device=dev)
            tb_o = torch.tensor(tbox_all[oth], dtype=torch.float32, device=dev)
            for name in ('head', 'head_notempl'):
                if name not in heads:
                    continue
                own = R.read_inputs(bank.feat, bank.id, idx_t, bank.feat, bank.id, idx_a, tb, pred, s0,
                                    lt[:, 0], lt[:, 1], lt[:, 2], la[:, 0], la[:, 1], la[:, 2])
                out = heads[name](*own, R.scalars(s0, age), use_template=(name == 'head'))
                pr, bx = R.decode(out, pred, s0)
                res[name][0][sel], res[name][1][sel] = pr.cpu().numpy(), bx.cpu().numpy()
                if name == 'head':
                    sh = R.read_inputs(bank.feat, bank.id, idx_t, bank.feat, bank.id, idx_o, tb_o, pred, s0,
                                       lt[:, 0], lt[:, 1], lt[:, 2], lo[:, 0], lo[:, 1], lo[:, 2])
                    out = heads['head'](*sh, R.scalars(s0, age), use_template=True)
                    pr, bx = R.decode(out, pred, s0)
                    res['head_shuffled'][0][sel], res['head_shuffled'][1][sel] = pr.cpu().numpy(), bx.cpu().numpy()
            del bank
        print(f'{seq_name}: {len(rows)} slots scored', flush=True)
    arrays = {}
    for arm, (pp, bb) in res.items():
        arrays[arm + '_p'], arrays[arm + '_box'] = pp, bb
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **arrays)
    repo = Path(__file__).resolve().parents[1]
    meta = {'slots': str(a.slots), 'slots_sha256': hashlib.sha256(a.slots.read_bytes()).hexdigest(), 'backbone_sha256': hashlib.sha256(a.backbone.read_bytes()).hexdigest(),
            'heads': {k: hashlib.sha256(Path(s.split('=')[1]).read_bytes()).hexdigest() for k, s in zip(heads, a.head)},
            'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'arms': sorted(res), 'seed': SEED}
    a.out.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
