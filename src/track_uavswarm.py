#!/usr/bin/env python3
"""Run the paper-based UAVS-MOT tracker over every UAVSwarm sequence."""

import _init_paths

from pathlib import Path

import torch

from datasets.dataset.jde import LoadImages
from opts import opts
from tracker.basetrack import BaseTrack
from tracker.multitracker import UAVSMOTTracker


def write_results(path, results):
    with path.open('w') as destination:
        for frame_id, tracks in results:
            for track in tracks:
                x, y, width, height = track.tlwh
                destination.write(
                    '{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},-1,-1,-1\n'.format(
                        frame_id, track.track_id, x, y, width, height, track.score))


def reset_sequence_state(tracker):
    tracker.tracked_stracks = []
    tracker.lost_stracks = []
    tracker.removed_stracks = []
    tracker.frame_id = 0
    BaseTrack._count = 0


def main(opt):
    dataset_root = Path(opt.uavs_dataset_root)
    result_root = Path(opt.uavs_result_dir)
    sequence_root = dataset_root / opt.uavs_split
    if not dataset_root.is_dir() or not sequence_root.is_dir():
        raise FileNotFoundError('invalid UAVSwarm root or split: ' + str(sequence_root))
    if not opt.load_model:
        raise ValueError('--load_model is required')
    result_root.mkdir(parents=True, exist_ok=True)
    tracker = UAVSMOTTracker(opt, frame_rate=30)
    sequence_dirs = sorted(path for path in sequence_root.glob('UAVSwarm-*') if path.is_dir())
    if not sequence_dirs:
        raise FileNotFoundError('no UAVSwarm sequences under ' + str(sequence_root))
    total_frames = 0
    for sequence_dir in sequence_dirs:
        image_dir = sequence_dir / 'img1'
        reset_sequence_state(tracker)
        results = []
        loader = LoadImages(str(image_dir), opt.img_size)
        for frame_id, (_, image, image_original) in enumerate(loader, start=1):
            blob = torch.from_numpy(image).to(opt.device).unsqueeze(0)
            results.append((frame_id, tracker.update(blob, image_original)))
        output = result_root / (sequence_dir.name + '.txt')
        write_results(output, results)
        total_frames += len(loader)
        print('{}: {} frames -> {}'.format(sequence_dir.name, len(loader), output))
    print('completed {} sequences and {} frames'.format(len(sequence_dirs), total_frames))


if __name__ == '__main__':
    main(opts().init())
