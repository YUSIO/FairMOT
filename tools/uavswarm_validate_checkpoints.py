#!/usr/bin/env python3
"""Select a trained FairMOT checkpoint on the frozen UAVSwarm validation split.

For every candidate epoch this driver runs the unmodified upstream JDETracker
(``src/track_uavswarm_fairmot.py``) over the frozen validation directories and
then the repository evaluator (``tools/evaluate_uavswarm_mot.py``: motmetrics
MOT fields plus TrackEval HOTA).  Per-epoch tracker outputs, evaluator outputs
and logs are preserved so the selection can be re-audited.

The official test split is deliberately out of scope: this driver is restricted
to the official train split, and its sequences are the sequence-disjoint
validation holdout of ``experiments/exp031_fairmot_uavswarmv1_staged_reproduction``
(video_id 1, 11, 20, 25, 32, 36 -> directories UAVSwarm-01, 21, 39, 49, 63, 71).
Checkpoint selection never reads test images or test ground truth.

Ranking follows the frozen rule: HOTA desc, IDF1 desc, MOTA desc, earlier epoch.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


RANKING_RULE = ['HOTA_desc', 'IDF1_desc', 'MOTA_desc', 'earlier_epoch']
MOT_COUNT_FIELDS = ['num_false_positives', 'num_misses', 'num_switches',
                    'num_objects', 'num_predictions']


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_epochs(value):
    try:
        epochs = tuple(sorted({int(item) for item in value.split(',') if item}))
    except ValueError as error:
        raise argparse.ArgumentTypeError('epochs must be comma-separated integers') from error
    if not epochs or any(epoch < 1 for epoch in epochs):
        raise argparse.ArgumentTypeError('epochs must be positive integers')
    return epochs


def parse_sequences(value):
    sequences = tuple(sorted({item for item in value.split(',') if item}))
    if not sequences or any(not item.startswith('UAVSwarm-') for item in sequences):
        raise argparse.ArgumentTypeError(
            'sequences must be comma-separated UAVSwarm-NN directory names')
    return sequences


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--checkpoint-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=parse_epochs,
                        default=parse_epochs('10,20,30,40,50,60,70,80,90'))
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--sequences', type=parse_sequences, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--iou-threshold', type=float, default=0.5)
    parser.add_argument('--input-width', type=int, default=1088)
    parser.add_argument('--input-height', type=int, default=608)
    return parser.parse_args()


def run(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as log:
        log.write('$ ' + ' '.join(str(item) for item in command) + '\n')
        log.flush()
        completed = subprocess.run([str(item) for item in command],
                                   stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode


def read_metrics(path):
    metrics = json.loads(path.read_text())
    overall = metrics['overall']
    row = {
        'HOTA': overall['hota']['HOTA'],
        'DetA': overall['hota']['DetA'],
        'AssA': overall['hota']['AssA'],
        'IDF1': overall['motmetrics']['idf1'],
        'MOTA': overall['motmetrics']['mota'],
    }
    for field in MOT_COUNT_FIELDS:
        row[field] = overall['motmetrics'][field]
    return row


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    tracker_script = repo_root / 'src' / 'track_uavswarm_fairmot.py'
    evaluator_script = repo_root / 'tools' / 'evaluate_uavswarm_mot.py'
    for script in (tracker_script, evaluator_script):
        if not script.is_file():
            raise FileNotFoundError(script)

    args.output_root.mkdir(parents=True, exist_ok=True)
    candidates, failures = [], []
    for epoch in args.epochs:
        checkpoint = args.checkpoint_dir / 'model_{}.pth'.format(epoch)
        epoch_root = args.output_root / 'epoch_{:03d}'.format(epoch)
        record = {'epoch': epoch, 'checkpoint': str(checkpoint), 'status': 'missing'}
        if not checkpoint.is_file():
            print('epoch {}: missing checkpoint {}'.format(epoch, checkpoint), flush=True)
            candidates.append(record)
            failures.append(epoch)
            continue
        record['checkpoint_sha256'] = sha256(checkpoint)
        tracks = epoch_root / 'tracks'
        metrics_path = epoch_root / 'metrics.json'
        tracker_status = run([
            args.python, tracker_script, 'mot', '--gpus', 0,
            '--load_model', checkpoint,
            '--input_w', args.input_width, '--input_h', args.input_height,
            '--uavs-dataset-root', args.dataset_root,
            '--uavs-split', 'train',
            '--uavs-sequences', ','.join(args.sequences),
            '--uavs-result-dir', tracks,
        ], epoch_root / 'tracker.log')
        record['tracker_exit_code'] = tracker_status
        record['tracks'] = str(tracks)
        if tracker_status != 0:
            record['status'] = 'tracker_failed'
            print('epoch {}: tracker failed with exit code {}'.format(epoch, tracker_status), flush=True)
            candidates.append(record)
            failures.append(epoch)
            continue
        evaluator_status = run([
            args.python, evaluator_script,
            '--dataset-root', args.dataset_root,
            '--split', 'train',
            '--sequences', ','.join(args.sequences),
            '--tracker-results', tracks,
            '--output', metrics_path,
            '--iou-threshold', args.iou_threshold,
        ], epoch_root / 'evaluator.log')
        record['evaluator_exit_code'] = evaluator_status
        record['metrics'] = str(metrics_path)
        if evaluator_status != 0 or not metrics_path.is_file():
            record['status'] = 'evaluator_failed'
            print('epoch {}: evaluator failed with exit code {}'.format(epoch, evaluator_status), flush=True)
            candidates.append(record)
            failures.append(epoch)
            continue
        record['status'] = 'completed'
        record.update(read_metrics(metrics_path))
        candidates.append(record)
        print('epoch {:3d}: HOTA {:8.4f} IDF1 {:8.4f} MOTA {:8.4f} (FP {} FN {} IDSW {})'.format(
            epoch, record['HOTA'], record['IDF1'], record['MOTA'],
            record['num_false_positives'], record['num_misses'], record['num_switches']), flush=True)

    completed = [record for record in candidates if record['status'] == 'completed']
    selected = None
    if completed:
        selected = sorted(completed, key=lambda record: (
            -record['HOTA'], -record['IDF1'], -record['MOTA'], record['epoch']))[0]
        print('selected epoch {}: HOTA {:.4f} IDF1 {:.4f} MOTA {:.4f}'.format(
            selected['epoch'], selected['HOTA'], selected['IDF1'], selected['MOTA']), flush=True)

    output = {
        'protocol': {
            'split': 'train',
            'sequences': list(args.sequences),
            'checkpoint_dir': str(args.checkpoint_dir),
            'epochs': list(args.epochs),
            'iou_threshold': args.iou_threshold,
            'input_size': [args.input_width, args.input_height],
            'official_test_access': 'not_accessed_by_checkpoint_selection',
        },
        'ranking_rule': RANKING_RULE,
        'complete': not failures,
        'failed_epochs': failures,
        'candidates': candidates,
        'selected': selected,
        'selected_epoch': selected['epoch'] if selected else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print('wrote ' + str(args.output), flush=True)
    if failures:
        print('incomplete sweep, failed epochs: {}'.format(failures), flush=True)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
