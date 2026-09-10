#!/usr/bin/env python3
"""Run the unmodified upstream JDETracker on selected UAVSwarm sequences."""

import argparse
from pathlib import Path

import _init_paths
import torch

from datasets.dataset.jde import LoadImages
from opts import opts
from tracker.basetrack import BaseTrack
from tracker.multitracker import JDETracker


def parse_sequence_ids(value):
    sequence_ids = tuple(sorted({int(item) for item in value.split(',') if item}))
    if not sequence_ids or any(sequence_id < 1 or sequence_id > 36 for sequence_id in sequence_ids):
        raise argparse.ArgumentTypeError('sequence IDs must be comma-separated integers in 1..36')
    return sequence_ids


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--uavs-dataset-root', type=Path, required=True)
    parser.add_argument('--uavs-split', choices=('train', 'test'), required=True)
    parser.add_argument('--uavs-sequences', type=parse_sequence_ids, required=True)
    parser.add_argument('--uavs-result-dir', type=Path, required=True)
    custom, fairmot_args = parser.parse_known_args()
    return custom, opts().init(fairmot_args)


def write_results(path, results, min_box_area):
    with path.open('w') as destination:
        for frame_id, tracks in results:
            for track in tracks:
                tlwh = track.tlwh
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] <= min_box_area or vertical:
                    continue
                destination.write(
                    '{},{},{:.6f},{:.6f},{:.6f},{:.6f},{:.6f},-1,-1,-1\n'.format(
                        frame_id, track.track_id, tlwh[0], tlwh[1], tlwh[2], tlwh[3], track.score))


def reset_sequence_state(tracker):
    tracker.tracked_stracks = []
    tracker.lost_stracks = []
    tracker.removed_stracks = []
    tracker.frame_id = 0
    BaseTrack._count = 0


def main():
    custom, opt = parse_args()
    if not opt.load_model:
        raise ValueError('--load_model is required')
    sequence_root = custom.uavs_dataset_root / custom.uavs_split
    if not sequence_root.is_dir():
        raise FileNotFoundError(sequence_root)
    custom.uavs_result_dir.mkdir(parents=True, exist_ok=True)
    tracker = JDETracker(opt, frame_rate=30)
    total_frames = 0
    for sequence_id in custom.uavs_sequences:
        sequence_dir = sequence_root / 'UAVSwarm-{:02d}'.format(sequence_id)
        image_dir = sequence_dir / 'img1'
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        reset_sequence_state(tracker)
        results = []
        loader = LoadImages(str(image_dir), opt.img_size)
        for frame_id, (_, image, image_original) in enumerate(loader, start=1):
            blob = torch.from_numpy(image).to(opt.device).unsqueeze(0)
            results.append((frame_id, tracker.update(blob, image_original)))
        output = custom.uavs_result_dir / (sequence_dir.name + '.txt')
        write_results(output, results, opt.min_box_area)
        total_frames += len(loader)
        print('{}: {} frames -> {}'.format(sequence_dir.name, len(loader), output))
    print('completed {} sequences and {} frames'.format(len(custom.uavs_sequences), total_frames))


if __name__ == '__main__':
    main()
