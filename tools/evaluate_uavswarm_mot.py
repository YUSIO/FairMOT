#!/usr/bin/env python3
"""Evaluate selected UAVSwarm sequences with MOT metrics and TrackEval HOTA."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import motmetrics as mm
import numpy as np
from trackeval.metrics import HOTA


if not hasattr(np, 'asfarray'):
    np.asfarray = lambda values, dtype=float: np.asarray(values, dtype=dtype)


COUNT_FIELDS = {'num_unique_objects', 'mostly_tracked', 'partially_tracked',
                'mostly_lost', 'num_false_positives', 'num_misses',
                'num_switches', 'num_fragmentations', 'num_objects',
                'num_predictions'}
RATE_FIELDS = {'idf1', 'idp', 'idr', 'recall', 'precision', 'mota'}
MOT_FIELDS = ['idf1', 'idp', 'idr', 'recall', 'precision',
              'num_unique_objects', 'mostly_tracked', 'partially_tracked',
              'mostly_lost', 'num_false_positives', 'num_misses',
              'num_switches', 'num_fragmentations', 'mota', 'num_objects',
              'num_predictions']


def sequence_ids(value):
    values = tuple(sorted({int(item) for item in value.split(',') if item}))
    if not values or any(value < 1 or value > 36 for value in values):
        raise argparse.ArgumentTypeError('sequence IDs must be comma-separated integers in 1..36')
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--split', choices=('train', 'test'), required=True)
    parser.add_argument('--sequence-ids', type=sequence_ids, required=True)
    parser.add_argument('--tracker-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    return parser.parse_args()


def read_mot_boxes(path, min_confidence):
    frames = defaultdict(list)
    with path.open() as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            fields = line.strip().split(',')
            if len(fields) < 6:
                raise ValueError('{}:{}: expected at least 6 MOT fields'.format(path, line_number))
            confidence = float(fields[6]) if len(fields) > 6 else 1.0
            if confidence >= min_confidence:
                frames[int(float(fields[0]))].append(
                    (int(float(fields[1])), *(float(value) for value in fields[2:6])))
    return frames


def iou_matrix(gt_boxes, tracker_boxes):
    if not gt_boxes or not tracker_boxes:
        return np.empty((len(gt_boxes), len(tracker_boxes)), dtype=float)
    gt = np.asarray([box[1:] for box in gt_boxes], dtype=float)
    tracker = np.asarray([box[1:] for box in tracker_boxes], dtype=float)
    gt_right, gt_bottom = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]
    tr_right, tr_bottom = tracker[:, 0] + tracker[:, 2], tracker[:, 1] + tracker[:, 3]
    intersection = (np.maximum(0.0, np.minimum(gt_right[:, None], tr_right[None, :]) -
                               np.maximum(gt[:, None, 0], tracker[None, :, 0])) *
                    np.maximum(0.0, np.minimum(gt_bottom[:, None], tr_bottom[None, :]) -
                               np.maximum(gt[:, None, 1], tracker[None, :, 1])))
    union = gt[:, None, 2] * gt[:, None, 3] + tracker[None, :, 2] * tracker[None, :, 3] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def trackeval_data(gt_frames, tracker_frames):
    gt_source_ids = sorted({box[0] for boxes in gt_frames.values() for box in boxes})
    tracker_source_ids = sorted({box[0] for boxes in tracker_frames.values() for box in boxes})
    gt_id_map = {source_id: index for index, source_id in enumerate(gt_source_ids)}
    tracker_id_map = {source_id: index for index, source_id in enumerate(tracker_source_ids)}
    gt_ids, tracker_ids, similarities = [], [], []
    for frame in sorted(set(gt_frames) | set(tracker_frames)):
        gt_boxes, tracker_boxes = gt_frames[frame], tracker_frames[frame]
        gt_ids.append(np.asarray([gt_id_map[box[0]] for box in gt_boxes], dtype=int))
        tracker_ids.append(np.asarray([tracker_id_map[box[0]] for box in tracker_boxes], dtype=int))
        similarities.append(iou_matrix(gt_boxes, tracker_boxes))
    return {'gt_ids': gt_ids, 'tracker_ids': tracker_ids,
            'similarity_scores': similarities, 'num_gt_ids': len(gt_source_ids),
            'num_tracker_ids': len(tracker_source_ids),
            'num_gt_dets': sum(len(boxes) for boxes in gt_frames.values()),
            'num_tracker_dets': sum(len(boxes) for boxes in tracker_frames.values())}


def serialise_mot_row(row):
    values = {}
    for field in MOT_FIELDS:
        value = float(row[field])
        if not np.isfinite(value):
            values[field] = None
        elif field in COUNT_FIELDS:
            values[field] = int(value)
        elif field in RATE_FIELDS:
            values[field] = round(100.0 * value, 6)
        else:
            values[field] = round(value, 6)
    return values


def serialise_hota(result, metric):
    summary = {field: round(100.0 * float(np.mean(result[field])), 6)
               for field in ('HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr',
                             'AssRe', 'AssPr', 'LocA', 'OWTA')}
    summary['curve_percent'] = {field: [round(100.0 * float(value), 6)
                                        for value in result[field]]
                                for field in ('HOTA', 'DetA', 'AssA')}
    summary['iou_thresholds'] = [round(float(value), 2) for value in metric.array_labels]
    return summary


def main():
    args = parse_args()
    mm.lap.default_solver = 'lap'
    accumulators, names, hota_by_sequence = [], [], {}
    for sequence_id in args.sequence_ids:
        name = 'UAVSwarm-{:02d}'.format(sequence_id)
        sequence_dir = args.dataset_root / args.split / name
        gt_path = sequence_dir / 'gt' / 'gt.txt'
        tracker_path = args.tracker_results / (name + '.txt')
        if not gt_path.is_file() or not tracker_path.is_file():
            raise FileNotFoundError('missing GT or result for ' + name)
        gt = mm.io.loadtxt(str(gt_path), fmt='mot15-2D', min_confidence=1)
        tracker = mm.io.loadtxt(str(tracker_path), fmt='mot15-2D', min_confidence=-1)
        accumulators.append(mm.utils.compare_to_groundtruth(gt, tracker, 'iou', distth=args.iou_threshold))
        names.append(name)
        hota_by_sequence[name] = HOTA().eval_sequence(trackeval_data(
            read_mot_boxes(gt_path, 1.0), read_mot_boxes(tracker_path, -1.0)))
    mot_summary = mm.metrics.create().compute_many(
        accumulators, names=names, metrics=MOT_FIELDS, generate_overall=True)
    hota_metric = HOTA()
    output = {
        'units': {'rates': 'percent', 'counts': 'events or detections'},
        'protocol': {'clear_and_identity_iou_threshold': args.iou_threshold,
                     'hota_implementation': 'TrackEval 1.1.0 HOTA over IoU thresholds 0.05 through 0.95',
                     'split': args.split, 'sequence_ids': list(args.sequence_ids)},
        'overall': {'motmetrics': serialise_mot_row(mot_summary.loc['OVERALL']),
                    'hota': serialise_hota(hota_metric.combine_sequences(hota_by_sequence), hota_metric)},
        'per_sequence': {name: {'motmetrics': serialise_mot_row(mot_summary.loc[name]),
                                'hota': serialise_hota(hota_by_sequence[name], hota_metric)}
                         for name in names},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
