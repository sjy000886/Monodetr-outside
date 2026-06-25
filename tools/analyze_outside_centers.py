#!/usr/bin/env python
"""Report outside projected-center statistics from KITTI dataset targets."""

import argparse
import collections
import os
import sys

import numpy as np
import yaml


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset


CLASS_NAMES = {0: "Pedestrian", 1: "Car", 2: "Cyclist"}
DIFFICULTY_NAMES = {1: "Easy", 2: "Moderate", 3: "Hard"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze projected 3D centers using dataset target generation"
    )
    parser.add_argument("--config", default="configs/monodetr.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument(
        "--with-augmentation",
        action="store_true",
        help="Keep random train augmentation enabled for one seeded pass",
    )
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--respect-writelist",
        action="store_true",
        help="Use the config writelist instead of analyzing all three KITTI classes",
    )
    return parser.parse_args()


def direction_name(center_px, width, height):
    left = center_px[0] < 0
    right = center_px[0] > width - 1
    top = center_px[1] < 0
    bottom = center_px[1] > height - 1
    if left and top:
        return "left_top"
    if right and top:
        return "right_top"
    if left and bottom:
        return "left_bottom"
    if right and bottom:
        return "right_bottom"
    if left:
        return "left"
    if right:
        return "right"
    if top:
        return "top"
    if bottom:
        return "bottom"
    raise ValueError("direction_name received an inside projected center")


def main():
    args = parse_args()
    with open(args.config, "r") as config_file:
        dataset_cfg = dict(yaml.load(config_file, Loader=yaml.Loader)["dataset"])
    dataset_cfg["use_outside_center_modeling"] = True
    if not args.respect_writelist:
        dataset_cfg["writelist"] = ["Car", "Pedestrian", "Cyclist"]
    split = args.split or dataset_cfg["train_split"]

    np.random.seed(args.seed)
    dataset = KITTI_Dataset(split=split, cfg=dataset_cfg)
    if not args.with_augmentation:
        dataset.data_augmentation = False

    sample_count = len(dataset)
    if args.max_samples is not None:
        sample_count = min(sample_count, args.max_samples)

    total_targets = 0
    outside_targets = 0
    directions = collections.Counter()
    class_counts = collections.Counter()
    difficulty_counts = collections.Counter()
    offsets = []
    max_reconstruction_error = 0.0
    max_box_reconstruction_error = 0.0
    significant_negative_boxes = 0
    floating_negative_boxes = 0
    negative_directions = collections.Counter()
    minimum_margin = float("inf")
    invalid_examples = []

    for item in range(sample_count):
        _, _, targets, _ = dataset[item]
        valid = targets["mask_2d"]
        total_targets += int(valid.sum())
        outside = valid & targets["outside_center_mask"]
        outside_targets += int(outside.sum())

        representative_boxes = targets["boxes_3d_representative"][valid]
        representative_px = targets["representative_center_px"][valid]
        boxes = targets["boxes"][valid]
        box_xyxy_px = np.stack(
            [
                (boxes[:, 0] - boxes[:, 2] / 2) * dataset.resolution[0],
                (boxes[:, 1] - boxes[:, 3] / 2) * dataset.resolution[1],
                (boxes[:, 0] + boxes[:, 2] / 2) * dataset.resolution[0],
                (boxes[:, 1] + boxes[:, 3] / 2) * dataset.resolution[1],
            ],
            axis=1,
        ) if len(boxes) else np.zeros((0, 4), dtype=np.float32)
        raw_margins = np.stack(
            [
                representative_px[:, 0] - box_xyxy_px[:, 0],
                box_xyxy_px[:, 2] - representative_px[:, 0],
                representative_px[:, 1] - box_xyxy_px[:, 1],
                box_xyxy_px[:, 3] - representative_px[:, 1],
            ],
            axis=1,
        ) if len(boxes) else np.zeros((0, 4), dtype=np.float32)
        if len(raw_margins):
            minimum_margin = min(minimum_margin, float(raw_margins.min()))
            significant = (raw_margins < -1e-6).any(axis=1)
            floating = (
                ((raw_margins < 0) & (raw_margins >= -1e-6)).any(axis=1)
                & ~significant
            )
            significant_negative_boxes += int(significant.sum())
            floating_negative_boxes += int(floating.sum())
            for direction_index, direction in enumerate(
                    ("left", "right", "top", "bottom")):
                negative_directions[direction] += int(
                    (raw_margins[:, direction_index] < -1e-6).sum()
                )

            valid_slots = np.flatnonzero(valid)
            for local_index in np.flatnonzero(significant):
                if len(invalid_examples) >= 20:
                    break
                slot = int(valid_slots[local_index])
                invalid_examples.append({
                    "item": item,
                    "class": CLASS_NAMES[int(targets["labels"][slot])],
                    "margins_px": raw_margins[local_index].tolist(),
                })

        if len(representative_boxes):
            reconstructed_box = np.stack(
                [
                    representative_boxes[:, 0] - representative_boxes[:, 2],
                    representative_boxes[:, 1] - representative_boxes[:, 4],
                    representative_boxes[:, 0] + representative_boxes[:, 3],
                    representative_boxes[:, 1] + representative_boxes[:, 5],
                ],
                axis=1,
            )
            expected_box = box_xyxy_px / np.array(
                [
                    dataset.resolution[0],
                    dataset.resolution[1],
                    dataset.resolution[0],
                    dataset.resolution[1],
                ],
                dtype=np.float32,
            )
            valid_rep_box = targets["representative_box_valid_mask"][valid]
            if valid_rep_box.any():
                max_box_reconstruction_error = max(
                    max_box_reconstruction_error,
                    float(np.max(np.abs(
                        reconstructed_box[valid_rep_box]
                        - expected_box[valid_rep_box]
                    ))),
                )

        reconstructed = (
            targets["representative_center"][valid]
            + targets["outside_center_offset"][valid]
        )
        expected = targets["projected_3d_center"][valid]
        if len(expected):
            max_reconstruction_error = max(
                max_reconstruction_error,
                float(np.max(np.abs(reconstructed - expected))),
            )

        for slot in np.flatnonzero(outside):
            center_px = targets["projected_3d_center_px"][slot]
            directions[
                direction_name(
                    center_px, dataset.resolution[0], dataset.resolution[1]
                )
            ] += 1
            class_counts[CLASS_NAMES[int(targets["labels"][slot])]] += 1
            difficulty = int(targets["object_difficulty"][slot])
            difficulty_counts[DIFFICULTY_NAMES.get(difficulty, str(difficulty))] += 1
            offsets.append(targets["outside_center_offset"][slot].copy())

    ratio = outside_targets / total_targets if total_targets else 0.0
    print("split: {}".format(split))
    print("samples scanned: {}".format(sample_count))
    print("augmentation enabled: {}".format(dataset.data_augmentation))
    print("total valid targets: {}".format(total_targets))
    print("outside projected-center targets: {}".format(outside_targets))
    print("outside ratio: {:.6%}".format(ratio))
    print("directions:")
    for name in (
        "left",
        "right",
        "top",
        "bottom",
        "left_top",
        "right_top",
        "left_bottom",
        "right_bottom",
    ):
        print("  {}: {}".format(name, directions[name]))
    print("classes:")
    for name in ("Car", "Pedestrian", "Cyclist"):
        print("  {}: {}".format(name, class_counts[name]))
    print("difficulty:")
    for name in ("Easy", "Moderate", "Hard"):
        print("  {}: {}".format(name, difficulty_counts[name]))

    if offsets:
        offsets = np.asarray(offsets, dtype=np.float64)
        l1 = np.linalg.norm(offsets, ord=1, axis=1)
        l2 = np.linalg.norm(offsets, ord=2, axis=1)
        print("normalized outside_center_offset:")
        print("  x min/max: {:.8f} / {:.8f}".format(offsets[:, 0].min(), offsets[:, 0].max()))
        print("  y min/max: {:.8f} / {:.8f}".format(offsets[:, 1].min(), offsets[:, 1].max()))
        print("  L1 mean: {:.8f}".format(l1.mean()))
        print("  L2 mean: {:.8f}".format(l2.mean()))
        print("  L2 p95: {:.8f}".format(np.percentile(l2, 95)))
        print("  L2 max: {:.8f}".format(l2.max()))
        zero_prediction_loss = np.log1p(np.abs(offsets)).sum(axis=1)
        print(
            "  zero-pred log1p loss min/mean/max: "
            "{:.8f} / {:.8f} / {:.8f}".format(
                zero_prediction_loss.min(),
                zero_prediction_loss.mean(),
                zero_prediction_loss.max(),
            )
        )
    else:
        print("normalized outside_center_offset: no outside targets")
    print(
        "max reconstruction error: {:.10g}".format(max_reconstruction_error)
    )
    print("representative box geometry:")
    print(
        "  max valid-box reconstruction error: {:.10g}".format(
            max_box_reconstruction_error
        )
    )
    print(
        "  significant negative-margin targets: {}".format(
            significant_negative_boxes
        )
    )
    print(
        "  floating-error negative-margin targets: {}".format(
            floating_negative_boxes
        )
    )
    print(
        "  minimum raw margin px: {}".format(
            "{:.8f}".format(minimum_margin)
            if minimum_margin != float("inf")
            else "n/a"
        )
    )
    for direction in ("left", "right", "top", "bottom"):
        print(
            "  negative {} margins: {}".format(
                direction, negative_directions[direction]
            )
        )
    if invalid_examples:
        print("  invalid examples:")
        for example in invalid_examples:
            print(
                "    item={item} class={class} margins_px={margins_px}".format(
                    **example
                )
            )


if __name__ == "__main__":
    main()
