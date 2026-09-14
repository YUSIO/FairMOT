#!/usr/bin/env python3
"""Regression against the immutable execution revision on synthetic/cached inputs."""
import ast
from collections import deque
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/lib'))
import numpy as np
import torch
from tracker import multitracker as current
from tracker.basetrack import BaseTrack, TrackState
from tracking_utils.kalman_filter import KalmanFilter
from diagnose_byte import config, tracker_without_model


def original_classes():
    path = Path(__file__).resolve().parents[1]
    source = subprocess.check_output(['git', 'show',
        '46b8fa9d8244611f2c8d7309fb967cb7e28165ef:src/lib/tracker/multitracker.py'], cwd=path, text=True)
    tree = ast.parse(source)
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    namespace = dict(vars(current))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-original>', 'exec'), namespace)
    return namespace


def rows(tracks):
    return np.asarray([[t.track_id, *t.tlwh, t.score] for t in tracks]).reshape(-1, 6)


def compare(frames):
    original = original_classes()
    opt = config()
    opt.uavs_low_recover_lost = False
    for name in ['JDETracker', 'UAVSMOTTracker']:
        histories = []
        for legacy in [True, False]:
            BaseTrack._count = 0
            cls = original[name] if legacy else getattr(current, name)
            tracker = tracker_without_model(cls, opt)
            output = []
            for dets, feats in frames:
                if not legacy:
                    tracks = tracker.update_detections(dets, feats)
                elif name == 'UAVSMOTTracker':
                    tracker._decode_detections = lambda *_: [original['STrack'](
                        original['STrack'].tlbr_to_tlwh(box[:4]), box[4], f.copy())
                        for box, f in zip(dets, feats) if box[4] > .1]
                    tracks = tracker.update(None, None)
                else:
                    # Supply predecoded network values to the untouched old update.
                    # Identity normalization/gather is bypassed ONLY at this adapter.
                    original['F'] = SimpleNamespace(normalize=lambda x, **_: x)
                    original['mot_decode'] = lambda *a, **k: (torch.tensor(dets), None)
                    original['_tranpose_and_gather_feat'] = lambda *a: torch.tensor(feats)[None]
                    tracker.model = lambda *_: [{'hm': torch.zeros(1), 'wh': None, 'id': None, 'reg': None}]
                    tracker.post_process = lambda *_: {1: dets.copy()}
                    tracks = tracker.update(SimpleNamespace(shape=(1,3,608,1088)),
                                            SimpleNamespace(shape=(608,1088,3)))
                output.append(rows(tracks).copy())
            histories.append(output)
        for frame, (a, b) in enumerate(zip(*histories)):
            assert a.shape == b.shape and np.array_equal(a,b), (name, frame, a,b)
        print(f'PASS {name}: {len(frames)} frames byte-identical numeric rows')


def paper_lifecycle_tests():
    def run(scores, paper=True):
        opt = config()
        opt.uavs_low_recover_lost = paper
        BaseTrack._count = 0
        tracker = tracker_without_model(current.UAVSMOTTracker, opt)
        history = []
        for score in scores:
            dets = np.empty((0,5)) if score is None else np.array([[10,10,30,30,score]])
            features = np.eye(len(dets),512,dtype=np.float32)
            before = dets.copy(), features.copy()
            history.append(rows(tracker.update_detections(dets,features)).copy())
            assert np.array_equal(dets,before[0]) and np.array_equal(features,before[1])
            assert len({t.track_id for t in tracker.lost_stracks}) == len(tracker.lost_stracks)
            assert all(t.state == TrackState.Lost for t in tracker.lost_stracks) if paper else True
        return tracker, history
    _, old = run([.8,None,.5,.5,.8], False)
    _, repaired = run([.8,None,.5,.5,.8])
    assert [len(v) for v in old] == [1,0,0,0,1]
    assert [len(v) for v in repaired] == [1,0,1,1,1]
    assert repaired[2][0,0] == repaired[0][0,0] and repaired[2][0,-1] == .5
    tracker, expired = run([.8]+[None]*30+[.8,.8])
    assert len(expired[-2]) == 0 and expired[-1][0,0] != expired[0][0,0]
    assert any(t.track_id == 1 for t in tracker.removed_stracks)
    _, boundary = run([.7,.7])
    assert all(len(v)==0 for v in boundary)
    run([.8]+[None]*40)
    print('PASS paper: lost-low recovery, score refresh, expiry before match, strict birth threshold, no duplicate lost, cache immutable')


def main():
    rng = np.random.default_rng(317)
    frames = []
    features = np.eye(4, 512, dtype=np.float32)
    for frame in range(120):
        boxes, feats = [], []
        for identity in range(4):
            if rng.random() < .2:
                continue
            x = 50 + identity*60 + frame*.7
            boxes.append([x, 60, x+20, 80, rng.choice([.05,.3,.5,.65,.75,.9])])
            feats.append(features[identity])
        frames.append((np.asarray(boxes,dtype=np.float32).reshape(-1,5),
                       np.asarray(feats,dtype=np.float32).reshape(-1,512)))
    compare(frames)
    paper_lifecycle_tests()
    if len(sys.argv) > 1:
        data = np.load(sys.argv[1])
        compare([(data['dets'][a:b], data['features'][a:b])
                 for a,b in zip(data['starts'][:-1],data['starts'][1:])])


if __name__ == '__main__':
    main()
