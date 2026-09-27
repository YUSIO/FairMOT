#!/usr/bin/env python3
"""Default JDETracker must reproduce the dcaef40 implementation row-for-row on a frozen cache."""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/lib'))
sys.path.insert(0, str(ROOT / 'tools'))
import numpy as np
import exp035_attribution as exp
from tracker.basetrack import BaseTrack


def legacy_module(commit):
    source = subprocess.check_output(['git', 'show', f'{commit}:src/lib/tracker/multitracker.py'],
                                     cwd=ROOT, text=True)
    path = Path(tempfile.mkdtemp()) / 'legacy_multitracker.py'
    path.write_text(source)
    import tracker  # noqa: F401  resolve the legacy file's relative imports against this package
    spec = importlib.util.spec_from_file_location('tracker._legacy_multitracker', path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = 'tracker'
    spec.loader.exec_module(module)
    return module


def rows(tracker_cls, opt, cache):
    BaseTrack._count = 0
    tracker = exp.bare_tracker(opt)
    tracker.__class__ = tracker_cls
    out = []
    for frame in range(1, len(cache['starts'])):
        a, b = cache['starts'][frame-1:frame+1]
        for t in tracker.update_detections(cache['dets'][a:b], cache['features'][a:b]):
            out.append((frame, t.track_id, *np.round(t.tlwh, 6), round(float(t.score), 6)))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--sequences', required=True)
    p.add_argument('--legacy-commit', default='dcaef40952e6cda664bf52bd917df5cf0b49ad3c')
    args = p.parse_args()
    legacy = legacy_module(args.legacy_commit).JDETracker
    from tracker.multitracker import JDETracker
    opt = exp.native_config()
    for seq, cache in exp.load_cache(args.cache, args.sequences):
        new, old = rows(JDETracker, opt, cache), rows(legacy, opt, cache)
        assert new == old, f'{seq}: default JDETracker diverged from {args.legacy_commit}'
        print(f'PASS {seq}: {len(new)} identical rows')
    # The ablation path must actually change the first-stage cost.
    iou = exp.native_config(['--first_stage_cost', 'iou'])
    seq, cache = next(exp.load_cache(args.cache, args.sequences.split(',')[0]))
    assert rows(JDETracker, iou, cache) != rows(JDETracker, opt, cache), 'iou variant is a no-op'
    print('PASS iou variant differs from embedding default')


if __name__ == '__main__':
    main()
