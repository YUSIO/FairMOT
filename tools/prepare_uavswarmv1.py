#!/usr/bin/env python3
"""Create FairMOT labels and image links from official UAVSwarm V1 train JSON.

Only ``annotations/train.json`` is read.  Test annotations are deliberately
outside this preparation path so model selection cannot depend on test GT.
"""

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path


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
    args = parser.parse_args()

    annotation_path = args.dataset_root / 'annotations' / 'train.json'
    with annotation_path.open() as source:
        data = json.load(source)
    images = {entry['id']: entry for entry in data['images']}
    annotations = defaultdict(list)
    for annotation in data['annotations']:
        if annotation.get('iscrowd', 0):
            continue
        annotations[annotation['image_id']].append(annotation)

    # UAVSwarm track IDs are sequence-local.  Map (video_id, track_id) to a
    # contiguous dataset-global identity range required by JointDataset.
    source_ids = sorted({(images[item['image_id']]['video_id'], item['track_id'])
                         for item in data['annotations'] if not item.get('iscrowd', 0)})
    identity_map = {source_id: index for index, source_id in enumerate(source_ids)}

    output_root = args.output_root
    image_root = output_root / 'images'
    label_root = output_root / 'labels_with_ids'
    list_root = output_root / 'lists'
    for directory in (image_root, label_root, list_root):
        directory.mkdir(parents=True, exist_ok=True)

    list_lines = []
    box_count = 0
    per_sequence = defaultdict(lambda: {'images': 0, 'boxes': 0, 'ids': set()})
    for image_id in sorted(images):
        image = images[image_id]
        relative = Path(image['file_name'])
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
            identity = identity_map[(image['video_id'], annotation['track_id'])]
            center_x = (x1 + x2) * 0.5 / image_width
            center_y = (y1 + y2) * 0.5 / image_height
            norm_width = (x2 - x1) / image_width
            norm_height = (y2 - y1) / image_height
            lines.append('0 {} {:.8f} {:.8f} {:.8f} {:.8f}\n'.format(
                identity, center_x, center_y, norm_width, norm_height))
            box_count += 1
            per_sequence[relative.parts[0]]['boxes'] += 1
            per_sequence[relative.parts[0]]['ids'].add(identity)
        destination_label.write_text(''.join(lines))
        list_lines.append(str(Path('images') / relative) + '\n')
        per_sequence[relative.parts[0]]['images'] += 1

    train_list = list_root / 'uavswarm.train'
    train_list.write_text(''.join(list_lines))
    summary = {
        'source_annotation': str(annotation_path),
        'source_annotation_sha256': sha256(annotation_path),
        'images': len(images),
        'boxes': box_count,
        'global_identities': len(identity_map),
        'identity_mapping': 'dense index over sorted (video_id, track_id)',
        'train_list': str(train_list),
        'per_sequence': {
            name: {'images': values['images'], 'boxes': values['boxes'],
                   'identities': len(values['ids'])}
            for name, values in sorted(per_sequence.items())
        },
    }
    (output_root / 'prepare_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
