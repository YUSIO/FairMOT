#!/usr/bin/env python3
"""Prepare a selected set of UAVSwarm V1 train sequences for upstream FairMOT.

Only ``annotations/train.json`` and the corresponding train images are read.
The output has the upstream FairMOT ``images``, ``labels_with_ids`` and list
layout.  The selected source sequences are mapped to a dense global Re-ID
class range; validation sequences are intentionally excluded.

Sequences are selected by annotation ``video_id`` (the dense 1..36 index of
``train.json``), while files and directories keep the dataset directory names
(``UAVSwarm-NN``, odd directories inside the official train split).  The
``video_id`` to directory mapping is written into the summary so that later
tracking and evaluation can address sequences by directory name.  A FairMOT
data-config JSON for ``src/train.py --data_cfg`` is written next to the list.
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


def parse_sequences(value):
    try:
        sequences = tuple(sorted({int(item) for item in value.split(',') if item}))
    except ValueError as error:
        raise argparse.ArgumentTypeError('sequence IDs must be comma-separated integers') from error
    if not sequences or any(sequence < 1 or sequence > 36 for sequence in sequences):
        raise argparse.ArgumentTypeError('sequence IDs must be within 1..36')
    return sequences


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--sequences', type=parse_sequences, required=True)
    parser.add_argument('--data-config-name', default='uavswarm_fairmot.json',
                        help='file name of the generated FairMOT data config')
    args = parser.parse_args()

    annotation_path = args.dataset_root / 'annotations' / 'train.json'
    with annotation_path.open() as source:
        data = json.load(source)
    selected = set(args.sequences)
    images = {entry['id']: entry for entry in data['images']
              if int(entry['video_id']) in selected}
    if not images:
        raise ValueError('no images for selected sequences')
    annotations = defaultdict(list)
    for annotation in data['annotations']:
        if annotation.get('iscrowd', 0) or annotation['image_id'] not in images:
            continue
        annotations[annotation['image_id']].append(annotation)

    source_ids = sorted({(int(images[item['image_id']]['video_id']), int(item['track_id']))
                         for items in annotations.values() for item in items})
    identity_map = {source_id: index for index, source_id in enumerate(source_ids)}

    image_root = args.output_root / 'images'
    label_root = args.output_root / 'labels_with_ids'
    list_root = args.output_root / 'lists'
    for directory in (image_root, label_root, list_root):
        directory.mkdir(parents=True, exist_ok=True)

    list_lines, box_count = [], 0
    sequence_directories = defaultdict(set)
    per_sequence = defaultdict(lambda: {'images': 0, 'boxes': 0, 'ids': set()})
    for image_id in sorted(images):
        image = images[image_id]
        relative = Path(image['file_name'])
        sequence_id = int(image['video_id'])
        sequence_directories[sequence_id].add(relative.parts[0])
        source_image = args.dataset_root / 'train' / relative
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        destination_image = image_root / relative
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        if destination_image.exists() or destination_image.is_symlink():
            if not destination_image.is_symlink() or destination_image.resolve() != source_image.resolve():
                raise FileExistsError('refusing to overwrite ' + str(destination_image))
        else:
            os.symlink(source_image, destination_image)

        destination_label = (label_root / relative).with_suffix('.txt')
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for annotation in annotations[image_id]:
            x, y, width, height = (float(value) for value in annotation['bbox'])
            image_width, image_height = float(image['width']), float(image['height'])
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(image_width, x + width), min(image_height, y + height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError('invalid clipped box for annotation {}'.format(annotation['id']))
            sequence_id = int(image['video_id'])
            identity = identity_map[(sequence_id, int(annotation['track_id']))]
            center_x = (x1 + x2) * 0.5 / image_width
            center_y = (y1 + y2) * 0.5 / image_height
            norm_width = (x2 - x1) / image_width
            norm_height = (y2 - y1) / image_height
            lines.append('0 {} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(
                identity, center_x, center_y, norm_width, norm_height))
            box_count += 1
            key = str(sequence_id)
            per_sequence[key]['boxes'] += 1
            per_sequence[key]['ids'].add(identity)
        destination_label.write_text(''.join(lines))
        list_lines.append(str(Path('images') / relative) + '\n')
        per_sequence[str(int(image['video_id']))]['images'] += 1

    train_list = list_root / 'uavswarm.train'
    train_list.write_text(''.join(list_lines))
    directories = {}
    for sequence_id, names in sorted(sequence_directories.items()):
        if len(names) != 1:
            raise ValueError('video_id {} spans directories {}'.format(sequence_id, sorted(names)))
        directories[str(sequence_id)] = sorted(names)[0]
    if set(directories) != {str(sequence) for sequence in args.sequences}:
        raise ValueError('video_id to directory mapping does not cover the selected sequences')

    data_config_path = args.output_root / args.data_config_name
    data_config_path.write_text(json.dumps({
        'root': str(args.output_root.resolve()),
        'train': {'uavswarm': str(train_list.resolve())},
    }, indent=2, sort_keys=True) + '\n')
    summary = {
        'source_annotation': str(annotation_path),
        'source_annotation_sha256': sha256(annotation_path),
        'sequences': list(args.sequences),
        'sequence_directories': directories,
        'images': len(images),
        'boxes': box_count,
        'global_identities': len(identity_map),
        'identity_mapping': 'dense index over selected (video_id, track_id)',
        'train_list': str(train_list),
        'data_config': str(data_config_path),
        'data_config_sha256': sha256(data_config_path),
        'per_sequence': {name: {'directory': directories[name], 'images': values['images'],
                                'boxes': values['boxes'], 'identities': len(values['ids'])}
                         for name, values in sorted(per_sequence.items(), key=lambda item: int(item[0]))},
    }
    (args.output_root / 'prepare_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
