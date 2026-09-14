#!/usr/bin/env python3
"""Frozen-cache BYTE diagnostics. GT is read only in the evaluation phase."""
import argparse
from collections import Counter
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
from tracker.multitracker import JDETracker, UAVSMOTTracker
from tracking_utils.kalman_filter import KalmanFilter
logging.disable(logging.DEBUG)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def config():
    return opts().init(['mot', '--gpus', '-1', '--reid_dim', '512',
                        '--use_coordinate_attention', '--ca_reduction', '32'])


def tracker_without_model(cls, opt):
    tracker = cls.__new__(cls)
    tracker.opt = opt
    tracker.tracked_stracks, tracker.lost_stracks, tracker.removed_stracks = [], [], []
    tracker.frame_id, tracker.det_thresh = 0, opt.conf_thres
    tracker.buffer_size = tracker.max_time_lost = 30
    tracker.max_per_image = opt.K
    tracker.kalman_filter = KalmanFilter()
    return tracker


def export(args):
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no implicit device fallback')
    torch.set_num_threads(4)
    opt = config()
    model = create_model(opt.arch, opt.heads, opt.head_conv,
                         use_coordinate_attention=True, ca_reduction=32)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['state_dict'], strict=True)
    model.to(args.device).eval()
    tracker = tracker_without_model(JDETracker, opt)
    tracker.model = model
    paths = sorted((args.dataset / args.split).glob('UAVSwarm-*'))
    if args.sequences:
        wanted = set(args.sequences.split(','))
        paths = [p for p in paths if p.name in wanted]
        assert {p.name for p in paths} == wanted
    manifest = {'phase': 'cache_export_diagnostic', 'split': args.split,
        'checkpoint_sha256': sha(args.checkpoint), 'device': args.device,
        'torch': torch.__version__, 'checkpoint_epoch': state['epoch'],
        'top_k': opt.K, 'feature_score_floor': .1, 'sequences': {},
        'purpose': 'train-seen diagnosis, not held-out performance'}
    write_json(args.output / 'cache_manifest.json', manifest)
    start = time.monotonic()
    for path in paths:
        loader = LoadImages(str(path / 'img1'), opt.img_size)
        loader.files = [p for p in loader.files if not Path(p).name.startswith('._')]
        loader.nF = len(loader.files)
        boxes, features, starts, raw_boxes, names, image_hashes = [], [], [0], [], [], []
        for frame, (image_path, image, original) in enumerate(loader, 1):
            with torch.inference_mode():
                blob = torch.from_numpy(image).unsqueeze(0).to(args.device)
                dets, feats = tracker.decode_arrays(blob, original)
            keep = dets[:, 4] > .1
            raw_boxes.append(dets)
            boxes.append(dets[keep])
            features.append(feats[keep])
            starts.append(starts[-1] + int(keep.sum()))
            names.append(Path(image_path).name)
            image_hashes.append(sha(image_path))
            if frame % 100 == 0:
                print(f'{path.name} {frame}/{len(loader)} frames; elapsed {time.monotonic()-start:.1f}s', flush=True)
            if args.max_frames and frame >= args.max_frames:
                break
        output = args.output / (path.name + '.npz')
        np.savez_compressed(output, dets=np.concatenate(boxes),
            features=np.concatenate(features), starts=np.array(starts),
            raw_dets=np.stack(raw_boxes), names=np.array(names), image_sha256=np.array(image_hashes))
        manifest['sequences'][path.name] = {'frames': len(names), 'rows': starts[-1],
            'sha256': sha(output)}
        write_json(args.output / 'cache_manifest.json', manifest)
        print(f'CACHE {path.name}: {len(names)} frames, {starts[-1]} rows, elapsed {time.monotonic()-start:.1f}s', flush=True)
    manifest['complete'] = True
    write_json(args.output / 'cache_manifest.json', manifest)


def read_gt(dataset, split, seq):
    result = {}
    for line in (dataset / split / seq / 'gt/gt.txt').read_text().splitlines():
        row = [float(x) for x in line.split(',')]
        if row[6] >= 1:
            result.setdefault(int(row[0]), []).append([int(row[1]), *row[2:6]])
    return result


def overlaps(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1, 4), np.asarray(b, dtype=float).reshape(-1, 4)
    inter = np.maximum(0., np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                       np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    area_a = np.maximum(0., a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = np.maximum(0., b[:, 2:] - b[:, :2]).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def audit_frame(gt, dets, raw, outputs, trace):
    """IoU one-to-one event locator; NOT the continuity-aware MOT evaluator."""
    from scipy.optimize import linear_sum_assignment
    g = np.asarray(gt, dtype=float).reshape(-1, 5)
    gboxes = g[:, 1:].copy()
    gboxes[:, 2:] += gboxes[:, :2]
    out_boxes = [t.tlbr.copy() for t in outputs]
    out_iou = overlaps(gboxes, out_boxes)
    matched_gt = set()
    if out_iou.size:
        eligible = out_iou >= .5
        # Maximize eligible pair count before maximizing IoU.
        rows, cols = linear_sum_assignment(-(eligible * (min(out_iou.shape) + 1) + out_iou * eligible))
        matched_gt = {int(r) for r, c in zip(rows, cols) if eligible[r, c]}
    det_iou, raw_iou = overlaps(gboxes, dets[:, :4]), overlaps(gboxes, raw[:, :4])
    stages = trace.get('stages', [])
    matched_dets = {}
    for stage in stages:
        for ti, di in stage['matches']:
            matched_dets[stage['det_indices'][di]] = (stage['stage'], stage['track_ids'][ti])
    pool = trace.get('pool', [])
    pool_boxes = [t['box'] for t in pool]
    pool_gt_iou = overlaps(gboxes, pool_boxes)
    events = []
    for gi in range(len(g)):
        if gi in matched_gt:
            continue
        candidates = np.flatnonzero(det_iou[gi] >= .5)
        raw_candidates = np.flatnonzero(raw_iou[gi] >= .5)
        flags = []
        if not len(raw_candidates):
            cause = 'no_topk_detection_iou50'
        elif not len(candidates):
            cause = 'all_correct_detections_score_at_most_01'
        elif any(int(di) in matched_dets for di in candidates):
            cause = 'correct_detection_matched_but_output_not_iou50'
        else:
            for di in candidates:
                score = float(dets[di, 4])
                near = np.flatnonzero(overlaps([dets[di, :4]], pool_boxes)[0] >= .2)
                if score <= .6:
                    lost_near = [pool[i]['id'] for i in near if pool[i]['state'] == 2]
                    eligible_ids = {i for s in stages if s['stage']=='low' for i in s['track_ids']}
                    if any(i not in eligible_ids for i in lost_near):
                        flags.append('lost_low_candidate_excluded')
                    if not len(near):
                        flags.append('low_no_overlapping_pool_track')
                if int(di) in trace.get('birth_rejected', []):
                    flags.append('birth_score_below_07')
                if int(di) in trace.get('births', []):
                    flags.append('birth_waiting_confirmation')
                if score > .6 and len(pool):
                    hi = next(s for s in stages if s['stage']=='high')
                    col = hi['det_indices'].index(int(di))
                    geo = np.array(trace['high_iou_cost'])[:, col]
                    fused = [row[col] for row in hi['costs']]
                    if not (geo <= .8).any():
                        flags.append('high_iou_gate_no_candidate')
                    elif all(v is None or v > .7 for v in fused):
                        flags.append('high_fused_cost_no_candidate')
            priority = ['lost_low_candidate_excluded', 'birth_score_below_07',
                        'birth_waiting_confirmation', 'high_iou_gate_no_candidate',
                        'high_fused_cost_no_candidate', 'low_no_overlapping_pool_track']
            cause = next((v for v in priority if v in flags), 'assignment_or_confirmation_unresolved')
        events.append({'gt_id': int(g[gi, 0]), 'cause': cause, 'flags': sorted(set(flags)),
            'correct_detection_indices': candidates.tolist(),
            'correct_scores': dets[candidates, 4].tolist(),
            'max_raw_score_at_iou50': float(raw[raw_candidates, 4].max()) if len(raw_candidates) else None,
            'max_pool_gt_iou': float(pool_gt_iou[gi].max()) if len(pool) else 0.})
    return events, matched_gt


def replay(args):
    opt = config()
    manifest = json.loads((args.cache / 'cache_manifest.json').read_text())
    if args.sequences:
        wanted = args.sequences.split(',')
        assert all(seq in manifest['sequences'] for seq in wanted)
        manifest['sequences'] = {seq: manifest['sequences'][seq] for seq in wanted}
    else:
        assert manifest.get('complete')
    write_json(args.output / 'cache_snapshot.json', manifest)
    variants = args.variants.split(',')
    summary = {'scope': 'diagnostic frame-IoU events, not formal MOT metrics', 'variants': {}}
    for variant in variants:
        cls = JDETracker if variant == 'native' else UAVSMOTTracker
        assert variant in ('native', 'legacy', 'lost_only', 'paper')
        opt.uavs_low_recover_lost = variant in ('lost_only', 'paper')
        opt.uavs_fix_lifecycle = variant == 'paper'
        folder = args.output / variant
        (folder / 'tracks').mkdir(parents=True)
        totals, per_sequence = Counter(), {}
        for seq, info in manifest['sequences'].items():
            path = args.cache / (seq + '.npz')
            assert sha(path) == info['sha256']
            cache = np.load(path)
            gt = read_gt(args.dataset, manifest['split'], seq)
            BaseTrack._count = 0
            tracker = tracker_without_model(cls, opt)
            counts = Counter()
            with (folder / 'tracks' / (seq + '.txt')).open('w') as tracks, \
                 (folder / (seq + '.jsonl')).open('w') as trace_file:
                for frame in range(1, len(cache['starts'])):
                    start, stop = cache['starts'][frame-1:frame+1]
                    dets, features = cache['dets'][start:stop], cache['features'][start:stop]
                    output = tracker.update_detections(dets, features)
                    rows = []
                    for t in output:
                        box = t.tlwh.copy()
                        tracks.write(f'{frame},{t.track_id},' + ','.join(f'{x:.6f}' for x in box) + f',{t.score:.6f},-1,-1,-1\n')
                        rows.append({'id': t.track_id, 'box': t.tlbr.tolist()})
                    trace = getattr(tracker, 'last_trace', {'frame': frame})
                    trace['outputs'] = rows
                    events, covered = audit_frame(gt.get(frame, []), dets, cache['raw_dets'][frame-1], output, trace) if variant != 'native' else ([], set())
                    trace['events'] = events
                    counts.update(e['cause'] for e in events)
                    counts['output_rows'] += len(output)
                    counts['gt_rows'] += len(gt.get(frame, []))
                    trace_file.write(json.dumps(trace, allow_nan=False) + '\n')
            per_sequence[seq] = dict(counts)
            totals.update(counts)
            print(f'REPLAY {variant} {seq}: {dict(counts)}', flush=True)
        summary['variants'][variant] = {'total': dict(totals), 'per_sequence': per_sequence}
        write_json(args.output / 'summary.json', summary)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=['export', 'replay'])
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--split', choices=['train', 'test'], default='train')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='mps')
    p.add_argument('--sequences')
    p.add_argument('--max-frames', type=int)
    p.add_argument('--cache', type=Path)
    p.add_argument('--variants', default='native,legacy,lost_only,paper')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    class Tee:
        def __init__(self, stream, file):
            self.stream, self.file = stream, file
        def write(self, value):
            self.stream.write(value)
            self.file.write(value)
            self.file.flush()
        def flush(self):
            self.stream.flush()
            self.file.flush()
    sys.stdout = Tee(sys.stdout, (args.output / 'stdout.log').open('w'))
    sys.stderr = Tee(sys.stderr, (args.output / 'stderr.log').open('w'))
    write_json(args.output / 'execution.json', {'argv': sys.argv, 'python': sys.executable,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip(),
        'status': 'running'})
    try:
        (export if args.phase == 'export' else replay)(args)
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
