#!/usr/bin/env python
"""Visualize bbox, projected center, representative center, and offset."""

import argparse
import math
import os
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.helpers.decode_helper import decode_projected_center


COLORS = {
    "bbox": (40, 170, 255),
    "bbox_center": (30, 210, 80),
    "representative": (255, 220, 30),
    "projected": (240, 45, 45),
    "decoded": (255, 70, 220),
    "boundary": (245, 245, 245),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render outside projected-center KITTI targets"
    )
    parser.add_argument("--config", default="configs/monodetr.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--output-dir", default="debug/outside_center_visualizations"
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument(
        "--indices",
        default=None,
        help="Optional comma-separated dataset item indices instead of category scan",
    )
    parser.add_argument(
        "--with-augmentation",
        action="store_true",
        help="Keep random train augmentation enabled for one seeded pass",
    )
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument(
        "--passes",
        type=int,
        default=20,
        help="Seeded augmentation passes used to collect enough edge examples",
    )
    parser.add_argument(
        "--respect-writelist",
        action="store_true",
        help="Use the config writelist instead of scanning all three KITTI classes",
    )
    return parser.parse_args()


def denormalize_image(inputs, dataset):
    image = inputs.transpose(1, 2, 0)
    image = (image * dataset.std + dataset.mean) * 255.0
    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))


def dashed_line(draw, start, end, fill, width=2, dash=10):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    length = np.linalg.norm(end - start)
    if length == 0:
        return
    direction = (end - start) / length
    distance = 0.0
    while distance < length:
        segment_end = min(distance + dash, length)
        draw.line(
            [
                tuple(start + direction * distance),
                tuple(start + direction * segment_end),
            ],
            fill=fill,
            width=width,
        )
        distance += 2 * dash


def arrow(draw, start, end, fill, width=3):
    draw.line([start, end], fill=fill, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    angle = math.atan2(dy, dx)
    head = 12
    for delta in (2.6, -2.6):
        point = (
            end[0] + head * math.cos(angle + delta),
            end[1] + head * math.sin(angle + delta),
        )
        draw.line([end, point], fill=fill, width=width)


def point(draw, xy, fill, radius=5):
    x, y = xy
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius],
        fill=fill,
        outline=(0, 0, 0),
    )


def direction_names(center, width, height):
    names = []
    if center[0] < 0:
        names.append("left")
    elif center[0] > width - 1:
        names.append("right")
    if center[1] < 0:
        names.append("top")
    elif center[1] > height - 1:
        names.append("bottom")
    return names or ["inside"]


def render(inputs, targets, slot, dataset, output_path):
    width, height = dataset.resolution
    image = denormalize_image(inputs, dataset)
    box = targets["boxes"][slot].astype(np.float64)
    box_px = np.array(
        [
            (box[0] - box[2] / 2) * width,
            (box[1] - box[3] / 2) * height,
            (box[0] + box[2] / 2) * width,
            (box[1] + box[3] / 2) * height,
        ]
    )
    bbox_center = targets["bbox_2d_center_px"][slot].astype(np.float64)
    projected = targets["projected_3d_center_px"][slot].astype(np.float64)
    representative = targets["representative_center_px"][slot].astype(np.float64)
    decoded_norm = decode_projected_center(
        targets["representative_center"][slot],
        targets["outside_center_offset"][slot],
    )
    decoded = decoded_norm.astype(np.float64) * dataset.resolution

    margin = 80.0
    min_x = min(0.0, box_px[0], projected[0], decoded[0]) - margin
    min_y = min(0.0, box_px[1], projected[1], decoded[1]) - margin
    max_x = max(width - 1.0, box_px[2], projected[0], decoded[0]) + margin
    max_y = max(height - 1.0, box_px[3], projected[1], decoded[1]) + margin
    scale = min(1.0, 2400.0 / max(max_x - min_x, max_y - min_y))

    def transform(xy):
        return (
            (float(xy[0]) - min_x) * scale,
            (float(xy[1]) - min_y) * scale,
        )

    canvas = Image.new(
        "RGB",
        (
            max(1, int(math.ceil((max_x - min_x) * scale))),
            max(1, int(math.ceil((max_y - min_y) * scale))),
        ),
        (35, 35, 35),
    )
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.BILINEAR,
    )
    canvas.paste(resized, tuple(map(int, transform((0, 0)))))
    draw = ImageDraw.Draw(canvas)

    image_box = [
        transform((0, 0)),
        transform((width - 1, height - 1)),
    ]
    draw.rectangle(image_box, outline=COLORS["boundary"], width=3)
    draw.rectangle(
        [transform(box_px[:2]), transform(box_px[2:])],
        outline=COLORS["bbox"],
        width=3,
    )
    dashed_line(
        draw,
        transform(bbox_center),
        transform(projected),
        COLORS["bbox_center"],
    )
    arrow(
        draw,
        transform(representative),
        transform(decoded),
        COLORS["decoded"],
    )
    point(draw, transform(bbox_center), COLORS["bbox_center"])
    point(draw, transform(representative), COLORS["representative"])
    point(draw, transform(projected), COLORS["projected"])
    point(draw, transform(decoded), COLORS["decoded"], radius=3)

    label = (
        "xb green | xI yellow | decoded magenta | GT xc red | "
        "offset=({:.4f}, {:.4f})"
    ).format(
        targets["outside_center_offset"][slot, 0],
        targets["outside_center_offset"][slot, 1],
    )
    draw.rectangle([8, 8, 850, 31], fill=(0, 0, 0))
    draw.text((12, 12), label, fill=(255, 255, 255))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.save(output_path)


def main():
    args = parse_args()
    with open(args.config, "r") as config_file:
        dataset_cfg = dict(yaml.load(config_file, Loader=yaml.Loader)["dataset"])
    dataset_cfg["use_outside_center_modeling"] = True
    if not args.respect_writelist:
        dataset_cfg["writelist"] = ["Car", "Pedestrian", "Cyclist"]
    split = args.split or dataset_cfg["train_split"]

    dataset = KITTI_Dataset(split=split, cfg=dataset_cfg)
    if not args.with_augmentation:
        dataset.data_augmentation = False

    requested_indices = None
    if args.indices:
        requested_indices = {
            int(value.strip()) for value in args.indices.split(",") if value.strip()
        }
    counts = {"left": 0, "right": 0, "top": 0, "bottom": 0, "inside": 0}

    pass_count = (
        1
        if requested_indices is not None or not dataset.data_augmentation
        else max(1, args.passes)
    )
    mandatory_categories = ("left", "right", "inside")
    for pass_index in range(pass_count):
        np.random.seed(args.seed + pass_index)
        for item in range(len(dataset)):
            if requested_indices is not None and item not in requested_indices:
                continue
            inputs, _, targets, info = dataset[item]
            valid_slots = np.flatnonzero(targets["mask_2d"])
            for slot in valid_slots:
                center = targets["projected_3d_center_px"][slot]
                categories = direction_names(
                    center, dataset.resolution[0], dataset.resolution[1]
                )
                for category in categories:
                    if requested_indices is None and counts[category] >= args.count:
                        continue
                    filename = (
                        "pass_{:02d}_item_{:04d}_img_{:06d}_slot_{:02d}.png"
                    ).format(pass_index, item, int(info["img_id"]), slot)
                    render(
                        inputs,
                        targets,
                        slot,
                        dataset,
                        os.path.join(args.output_dir, category, filename),
                    )
                    counts[category] += 1

            if requested_indices is not None and item >= max(requested_indices):
                break
            if requested_indices is None and all(
                counts[category] >= args.count
                for category in mandatory_categories
            ):
                break
        if requested_indices is None and all(
            counts[category] >= args.count for category in mandatory_categories
        ):
            break

    print("output directory: {}".format(os.path.abspath(args.output_dir)))
    for category, count in counts.items():
        print("{}: {}".format(category, count))


if __name__ == "__main__":
    main()
