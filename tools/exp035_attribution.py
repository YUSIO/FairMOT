#!/usr/bin/env python3
"""Exp035: attribute FairMOT tracking to detector, appearance cue and associator.

Phases (GT is never read here; evaluation uses tools/evaluate_uavswarm_mot.py):
  export     native FairMOT forward pass -> frozen per-sequence boxes + embeddings
  replay     JDETracker variants on the frozen cache (no inference)
  bytedets   write the same frozen boxes as MOT det.txt for the ByteTrack fork
"""
import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/lib'))
logging.getLogger().setLevel(logging.WARNING)
import numpy as np
import torch
from datasets.dataset.jde import LoadImages
from models.model import create_model
from opts import opts
from tracker.basetrack import BaseTrack
from tracker.multitracker import JDETracker
from tracking_utils.kalman_filter import KalmanFilter
logging.disable(logging.DEBUG)

SCORE_FLOOR = .1  # equals ByteTrack's low-score lower bound (scores > 0.1)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def native_config(extra=()):
    # Native FairMOT: DLA-34, 128-d CE Re-ID head, no coordinate attention.
    return opts().init(['mot', '--gpus', '-1', *extra])


def bare_tracker(opt):
    tracker = JDETracker.__new__(JDETracker)
    tracker.opt = opt
    tracker.tracked_stracks, tracker.lost_stracks, tracker.removed_stracks = [], [], []
    tracker.frame_id, tracker.det_thresh = 0, opt.conf_thres
    tracker.buffer_size = tracker.max_time_lost = int(30 / 30.0 * opt.track_buffer)
    tracker.max_per_image = opt.K
    tracker.kalman_filter = KalmanFilter()
    return tracker


def selected(paths, sequences):
    if not sequences:
        return paths
    wanted = set(sequences.split(','))
    paths = [p for p in paths if p.name in wanted]
    assert {p.name for p in paths} == wanted, 'missing requested sequences'
    return paths


def export(args):
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no implicit device fallback')
    torch.set_num_threads(4)
    opt = native_config()
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    weights = {k[7:] if k.startswith('module.') else k: v for k, v in state['state_dict'].items()}
    # The ImageNet DLA classifier is saved with the checkpoint but is not on the forward path.
    unused = sorted(k for k in weights if k.startswith('base.fc.'))
    assert unused == ['base.fc.bias', 'base.fc.weight'], unused
    for key in unused:
        del weights[key]
    model.load_state_dict(weights, strict=True)
    model.to(args.device).eval()
    tracker = bare_tracker(opt)
    tracker.model = model
    paths = selected(sorted((args.dataset / args.split).glob('UAVSwarm-*')), args.sequences)
    manifest = {'phase': 'cache_export', 'split': args.split, 'checkpoint': str(args.checkpoint),
                'checkpoint_sha256': sha(args.checkpoint), 'checkpoint_epoch': state.get('epoch'),
                'device': args.device, 'torch': torch.__version__, 'heads': opt.heads,
                'input_wh': list(opt.img_size), 'top_k': opt.K, 'score_floor': SCORE_FLOOR,
                'dropped_unused_keys': unused,
                'sequences': {}}
    write_json(args.output / 'cache_manifest.json', manifest)
    start = time.monotonic()
    for path in paths:
        loader = LoadImages(str(path / 'img1'), opt.img_size)
        loader.files = [p for p in loader.files if not Path(p).name.startswith('._')]
        loader.nF = len(loader.files)
        boxes, features, starts, names, image_hashes = [], [], [0], [], []
        for frame, (image_path, image, original) in enumerate(loader, 1):
            with torch.inference_mode():
                blob = torch.from_numpy(image).unsqueeze(0).to(args.device)
                dets, feats = tracker.decode_arrays(blob, original)
            keep = dets[:, 4] > SCORE_FLOOR
            boxes.append(dets[keep])
            features.append(feats[keep])
            starts.append(starts[-1] + int(keep.sum()))
            names.append(Path(image_path).name)
            image_hashes.append(sha(image_path))
            if frame % 200 == 0:
                print(f'{path.name} {frame}/{len(loader)}; elapsed {time.monotonic()-start:.1f}s', flush=True)
        output = args.output / (path.name + '.npz')
        np.savez_compressed(output, dets=np.concatenate(boxes), features=np.concatenate(features),
                            starts=np.array(starts), names=np.array(names),
                            image_sha256=np.array(image_hashes))
        manifest['sequences'][path.name] = {'frames': len(names), 'rows': starts[-1], 'sha256': sha(output)}
        write_json(args.output / 'cache_manifest.json', manifest)
        print(f'CACHE {path.name}: {len(names)} frames, {starts[-1]} rows, '
              f'elapsed {time.monotonic()-start:.1f}s', flush=True)
    manifest['complete'] = True
    write_json(args.output / 'cache_manifest.json', manifest)


def load_cache(cache_dir, sequences):
    manifest = json.loads((cache_dir / 'cache_manifest.json').read_text())
    assert manifest.get('complete'), 'incomplete cache'
    names = sequences.split(',') if sequences else list(manifest['sequences'])
    for seq in names:
        path = cache_dir / (seq + '.npz')
        assert sha(path) == manifest['sequences'][seq]['sha256'], 'cache hash mismatch: ' + seq
        with np.load(path) as archive:
            yield seq, {k: archive[k] for k in ('starts', 'dets', 'features')}


VARIANTS = {
    # name: (first_stage_cost, first_stage_thresh, permute_embeddings)
    'native': ('embedding', .4, False),
    'iou_t04': ('iou', .4, False),
    'iou_t08': ('iou', .8, False),
    'perm_emb': ('embedding', .4, True),
}


def replay(args):
    variants = args.variants.split(',')
    assert all(v in VARIANTS for v in variants), variants
    summary = {'cache': str(args.cache), 'seed': args.seed, 'output_filter': 'none', 'variants': {}}
    for variant in variants:
        cost, thresh, permute = VARIANTS[variant]
        opt = native_config(['--first_stage_cost', cost, '--first_stage_thresh', str(thresh)])
        folder = args.output / variant
        folder.mkdir(parents=True)
        rows_total = 0
        for seq, cache in load_cache(args.cache, args.sequences):
            rng = np.random.default_rng([args.seed, int(seq.split('-')[1])])
            BaseTrack._count = 0
            tracker = bare_tracker(opt)
            with (folder / (seq + '.txt')).open('w') as out:
                for frame in range(1, len(cache['starts'])):
                    start, stop = cache['starts'][frame-1:frame+1]
                    dets = cache['dets'][start:stop]
                    features = cache['features'][start:stop]
                    if permute and len(features) > 1:
                        # Null control: same embedding distribution, identity content destroyed.
                        features = features[rng.permutation(len(features))]
                    for t in tracker.update_detections(dets, features):
                        x, y, w, h = t.tlwh.copy()
                        out.write(f'{frame},{t.track_id},{x:.6f},{y:.6f},{w:.6f},{h:.6f},{t.score:.6f},-1,-1,-1\n')
                        rows_total += 1
            print(f'REPLAY {variant} {seq} done', flush=True)
        summary['variants'][variant] = {'first_stage_cost': cost, 'first_stage_thresh': thresh,
                                        'permute_embeddings': permute, 'conf_thres': opt.conf_thres,
                                        'track_buffer': opt.track_buffer, 'output_rows': rows_total,
                                        'track_sha256': {p.name: sha(p) for p in sorted(folder.glob('*.txt'))}}
        write_json(args.output / 'replay_summary.json', summary)


def bytedets(args):
    written = {}
    for seq, cache in load_cache(args.cache, args.sequences):
        folder = args.output / seq
        folder.mkdir(parents=True)
        with (folder / 'det.txt').open('w') as out:
            for frame in range(1, len(cache['starts'])):
                start, stop = cache['starts'][frame-1:frame+1]
                for x1, y1, x2, y2, score in cache['dets'][start:stop, :5]:
                    out.write(f'{frame},-1,{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},{score:.6f},-1,-1,-1\n')
        written[seq] = sha(folder / 'det.txt')
    write_json(args.output / 'bytedets_manifest.json', {'cache': str(args.cache), 'det_sha256': written,
               'format': 'frame,-1,x,y,w,h,score,-1,-1,-1 original-image tlwh; rows with score > 0.1'})


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['export', 'replay', 'bytedets'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path)
    p.add_argument('--split', choices=['train', 'test'])
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    p.add_argument('--sequences')
    p.add_argument('--cache', type=Path)
    p.add_argument('--variants', default='native,iou_t04,iou_t08,perm_emb')
    p.add_argument('--seed', type=int, default=35)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)  # never reuse an output directory
    repo = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'],
                                    cwd=repo, text=True).strip()
    write_json(args.output / 'execution.json', {'argv': sys.argv, 'python': sys.executable,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'tracked_dirty': bool(dirty), 'status': 'running'})
    try:
        {'export': export, 'replay': replay, 'bytedets': bytedets}[args.phase](args)
    except BaseException:
        traceback.print_exc()
        (args.output / 'exit_code.txt').write_text('1\n')
        raise
    (args.output / 'exit_code.txt').write_text('0\n')
    execution = json.loads((args.output / 'execution.json').read_text())
    execution['status'] = 'completed'
    write_json(args.output / 'execution.json', execution)


if __name__ == '__main__':
    main()
