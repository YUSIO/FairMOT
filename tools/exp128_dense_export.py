#!/usr/bin/env python3
"""Exp128: dense peak export of a frozen FairMOT model (forward pass only, nothing is trained).

Same checkpoint loading, input size, 3 x 3 max-pool peaks, ltrb decoding and original-image coordinates as the Exp035
cache exporter (tools/exp035_attribution.py export); only the score floor is lowered from 0.1 to 0.01 and top_k raised
from 500 to 2000. Raw peaks are saved (no NMS); each keeps its L2-normalised 128-d Re-ID embedding.

Output layout equals the Exp035 caches: <seq>.npz with dets (n, 5) = x1, y1, x2, y2, score; features (n, 128);
starts (frames + 1); names; image_sha256; plus cache_manifest.json, execution.json, exit_code.txt.
"""
import argparse
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch

from exp035_attribution import bare_tracker, native_config, selected, sha, write_json
from datasets.dataset.jde import LoadImages
from models.model import create_model

SCORE_FLOOR = .01
TOP_K = 2000


def load_model(checkpoint, device):
    opt = native_config(['--K', str(TOP_K)])
    model = create_model(opt.arch, opt.heads, opt.head_conv)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = {k[7:] if k.startswith('module.') else k: v for k, v in state['state_dict'].items()}
    unused = sorted(k for k in weights if k.startswith('base.fc.'))
    assert unused == ['base.fc.bias', 'base.fc.weight'], unused
    for key in unused:
        del weights[key]
    model.load_state_dict(weights, strict=True)
    return opt, model.to(device).eval(), state, unused


def export(args):
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; no implicit device fallback')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no implicit device fallback')
    torch.set_num_threads(4)
    opt, model, state, unused = load_model(args.checkpoint, args.device)
    tracker = bare_tracker(opt)
    tracker.model = model
    paths = selected(sorted((args.dataset / args.split).glob('UAVSwarm-*')), args.sequences)
    manifest = {'phase': 'dense_peak_export', 'split': args.split, 'checkpoint': str(args.checkpoint),
                'checkpoint_sha256': sha(args.checkpoint), 'checkpoint_epoch': state.get('epoch'),
                'device': args.device, 'torch': torch.__version__, 'heads': opt.heads,
                'input_wh': list(opt.img_size), 'top_k': opt.K, 'score_floor': SCORE_FLOOR,
                'nms': 'none (3x3 max-pool peaks only)', 'max_frames': args.max_frames, 'dropped_unused_keys': unused, 'sequences': {}}
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
            if args.max_frames and frame >= args.max_frames:
                break
            if frame % 200 == 0:
                print(f'{path.name} {frame}/{len(loader)}; elapsed {time.monotonic()-start:.1f}s', flush=True)
        output = args.output / (path.name + '.npz')
        np.savez_compressed(output, dets=np.concatenate(boxes), features=np.concatenate(features),
                            starts=np.array(starts), names=np.array(names), image_sha256=np.array(image_hashes))
        saturated = int(sum(1 for a, b in zip(starts[:-1], starts[1:]) if b - a >= TOP_K))
        manifest['sequences'][path.name] = {'frames': len(names), 'rows': starts[-1], 'sha256': sha(output),
                                            'frames_at_top_k': saturated}
        write_json(args.output / 'cache_manifest.json', manifest)
        print(f'DENSE {path.name}: {len(names)} frames, {starts[-1]} rows, {saturated} frames at top_k, '
              f'elapsed {time.monotonic()-start:.1f}s', flush=True)
    manifest['complete'] = True
    write_json(args.output / 'cache_manifest.json', manifest)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'test'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cuda')
    p.add_argument('--sequences')
    p.add_argument('--max-frames', type=int, help='smoke tests only; formal runs export whole sequences')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)  # never reuse an output directory
    repo = Path(__file__).resolve().parents[1]
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo, text=True).strip()
    write_json(args.output / 'execution.json', {'argv': sys.argv, 'python': sys.executable,
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        'tracked_dirty': bool(dirty), 'status': 'running'})
    try:
        export(args)
    except BaseException:
        traceback.print_exc()
        (args.output / 'exit_code.txt').write_text('1\n')
        raise
    (args.output / 'exit_code.txt').write_text('0\n')
    execution = __import__('json').loads((args.output / 'execution.json').read_text())
    execution['status'] = 'completed'
    write_json(args.output / 'execution.json', execution)


if __name__ == '__main__':
    main()
