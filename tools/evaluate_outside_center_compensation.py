#!/usr/bin/env python
"""Evaluate center-offset compensation on deterministic cropped KITTI samples."""

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter

os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")

import numpy as np
import torch
import yaml
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.outside_center_utils import compute_boundary_intersection
from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from utils import box_ops


DIFFICULTY_NAMES = {1: "Easy", 2: "Moderate", 3: "Hard"}
PIXEL_SCALE = np.array([1280.0, 384.0], dtype=np.float64)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare representative-center and compensated-center errors on "
            "deterministic crop augmentations that create outside-center GTs."
        )
    )
    parser.add_argument(
        "--config", default="configs/eval_outside_D_full_legacy_ry.yaml"
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--score-threshold", type=float, default=0.2)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default="debug/outside_center_compensation_epoch159",
    )
    return parser.parse_args()


def pairwise_iou(boxes1, boxes2):
    """Return a robust pairwise IoU matrix for xyxy boxes."""
    boxes1 = np.asarray(boxes1, dtype=np.float64)
    boxes2 = np.asarray(boxes2, dtype=np.float64)
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float64)
    left_top = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_size = np.clip(right_bottom - left_top, 0.0, None)
    intersection = intersection_size[:, :, 0] * intersection_size[:, :, 1]
    size1 = np.clip(boxes1[:, 2:] - boxes1[:, :2], 0.0, None)
    size2 = np.clip(boxes2[:, 2:] - boxes2[:, :2], 0.0, None)
    area1 = size1[:, 0] * size1[:, 1]
    area2 = size2[:, 0] * size2[:, 1]
    union = area1[:, None] + area2[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def center_error_metrics(
        predicted_representative,
        predicted_offset,
        target_center,
        target_offset,
):
    predicted_representative = np.asarray(
        predicted_representative, dtype=np.float64
    )
    predicted_offset = np.asarray(predicted_offset, dtype=np.float64)
    target_center = np.asarray(target_center, dtype=np.float64)
    target_offset = np.asarray(target_offset, dtype=np.float64)
    predicted_full = predicted_representative + predicted_offset

    representative_error_xy = (
        predicted_representative - target_center
    ) * PIXEL_SCALE
    full_error_xy = (predicted_full - target_center) * PIXEL_SCALE
    predicted_offset_px = predicted_offset * PIXEL_SCALE
    target_offset_px = target_offset * PIXEL_SCALE
    offset_error_px = predicted_offset_px - target_offset_px

    target_norm = np.linalg.norm(target_offset_px)
    predicted_norm = np.linalg.norm(predicted_offset_px)
    cosine = np.nan
    if target_norm > 1e-8 and predicted_norm > 1e-8:
        cosine = float(
            np.dot(target_offset_px, predicted_offset_px)
            / (target_norm * predicted_norm)
        )

    representative_error = float(np.linalg.norm(representative_error_xy))
    full_error = float(np.linalg.norm(full_error_xy))
    return {
        "representative_error_px": representative_error,
        "full_error_px": full_error,
        "error_delta_px": full_error - representative_error,
        "representative_error_x_px": float(abs(representative_error_xy[0])),
        "representative_error_y_px": float(abs(representative_error_xy[1])),
        "full_error_x_px": float(abs(full_error_xy[0])),
        "full_error_y_px": float(abs(full_error_xy[1])),
        "target_offset_norm_px": float(target_norm),
        "predicted_offset_norm_px": float(predicted_norm),
        "offset_error_px": float(np.linalg.norm(offset_error_px)),
        "offset_cosine": cosine,
        "improved": bool(full_error < representative_error),
    }


def summarize_records(records, match_iou):
    total = len(records)
    assigned = [record for record in records if record["assigned"]]
    matched = [
        record
        for record in assigned
        if record["bbox_iou"] >= match_iou
    ]
    summary = {
        "targets": total,
        "assigned": len(assigned),
        "matched_iou_0.3": sum(
            record["bbox_iou"] >= 0.3 for record in assigned
        ),
        "matched_iou_0.5": sum(
            record["bbox_iou"] >= 0.5 for record in assigned
        ),
        "matched_iou_0.7": sum(
            record["bbox_iou"] >= 0.7 for record in assigned
        ),
        "metric_match_iou": match_iou,
        "metric_matches": len(matched),
    }
    if not matched:
        return summary

    metric_names = (
        "representative_error_px",
        "full_error_px",
        "error_delta_px",
        "target_offset_norm_px",
        "predicted_offset_norm_px",
        "offset_error_px",
    )
    for name in metric_names:
        values = np.asarray([record[name] for record in matched])
        summary[name + "_mean"] = float(values.mean())
        summary[name + "_median"] = float(np.median(values))
        summary[name + "_p90"] = float(np.percentile(values, 90))
    cosines = np.asarray(
        [
            record["offset_cosine"]
            for record in matched
            if np.isfinite(record["offset_cosine"])
        ]
    )
    summary["offset_cosine_mean"] = (
        float(cosines.mean()) if len(cosines) else None
    )
    summary["improved"] = sum(record["improved"] for record in matched)
    summary["improved_fraction"] = summary["improved"] / len(matched)
    return summary


def summarize_unique_source_objects(records, match_iou):
    all_keys = {
        (record["image_id"], record["target_slot"]) for record in records
    }
    matched = [
        record
        for record in records
        if record["assigned"] and record["bbox_iou"] >= match_iou
    ]
    grouped = {}
    for record in matched:
        key = (record["image_id"], record["target_slot"])
        grouped.setdefault(key, []).append(record)

    aggregate_records = []
    metric_names = (
        "representative_error_px",
        "full_error_px",
        "error_delta_px",
        "target_offset_norm_px",
        "predicted_offset_norm_px",
        "offset_error_px",
        "offset_cosine",
    )
    for group in grouped.values():
        aggregate = {
            "assigned": True,
            "bbox_iou": 1.0,
        }
        for name in metric_names:
            values = np.asarray(
                [
                    record[name]
                    for record in group
                    if np.isfinite(record[name])
                ]
            )
            aggregate[name] = (
                float(values.mean()) if len(values) else float("nan")
            )
        aggregate["improved"] = aggregate["error_delta_px"] < 0
        aggregate_records.append(aggregate)

    summary = summarize_records(aggregate_records, match_iou=0.0)
    summary["source_objects_total"] = len(all_keys)
    summary["source_objects_matched"] = len(grouped)
    summary["instance_match_iou"] = match_iou
    return summary


def augmentation_seed(base_seed, pass_index, item, dataset_size):
    return base_seed + pass_index * dataset_size + item


def sample_crop_parameters(seed, image_size, scale, shift):
    rng = np.random.RandomState(seed)
    rng.random_sample()  # random_flip draw; the benchmark fixes probability to 0.
    rng.random_sample()  # random_crop draw; the benchmark fixes probability to 1.
    crop_scale = np.clip(
        rng.randn() * scale + 1.0, 1.0 - scale, 1.0 + scale
    )
    crop_size = image_size * crop_scale
    center = image_size.astype(np.float64) / 2.0
    center[0] += image_size[0] * np.clip(
        rng.randn() * shift, -2.0 * shift, 2.0 * shift
    )
    center[1] += image_size[1] * np.clip(
        rng.randn() * shift, -2.0 * shift, 2.0 * shift
    )
    return center, crop_size


def encoded_outside_slots(dataset, objects, calib, transform):
    width, height = dataset.resolution
    outside_slots = []
    for slot, obj in enumerate(objects[:dataset.max_objs]):
        if obj.cls_type not in dataset.writelist:
            continue
        if obj.level_str == "UnKnown" or obj.pos[-1] < 2 or obj.pos[-1] > 65:
            continue

        bbox = obj.box2d.copy()
        bbox[:2] = affine_transform(bbox[:2], transform)
        bbox[2:] = affine_transform(bbox[2:], transform)
        bbox_center = np.array(
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
            dtype=np.float32,
        )
        center_3d = (obj.pos + [0, -obj.h / 2.0, 0]).reshape(-1, 3)
        projected_center, _ = calib.rect_to_img(center_3d)
        projected_center = affine_transform(projected_center[0], transform)
        outside = (
            projected_center[0] < 0
            or projected_center[0] > width - 1
            or projected_center[1] < 0
            or projected_center[1] > height - 1
        )
        if not outside:
            continue
        visible = (
            bbox[2] > bbox[0]
            and bbox[3] > bbox[1]
            and bbox[2] > 0
            and bbox[0] < width - 1
            and bbox[3] > 0
            and bbox[1] < height - 1
        )
        if not visible:
            continue
        try:
            compute_boundary_intersection(
                bbox_center, projected_center, width, height
            )
        except ValueError:
            continue
        if obj.trucation <= 0.5 and obj.occlusion <= 2:
            outside_slots.append(slot)
    return outside_slots


def find_augmented_samples(dataset, passes, base_seed, scale, shift):
    selected = []
    standard_outside = 0
    standard_images = 0
    dataset_size = len(dataset)
    for item in tqdm(
            range(dataset_size), desc="Scanning deterministic crop geometry"):
        image_id = int(dataset.idx_list[item])
        image_path = os.path.join(dataset.image_dir, "{:06d}.png".format(image_id))
        with Image.open(image_path) as image:
            image_size = np.asarray(image.size, dtype=np.float64)
        objects = dataset.get_label(image_id)
        calib = dataset.get_calib(image_id)

        standard_transform = get_affine_transform(
            image_size / 2.0, image_size, 0, dataset.resolution
        )
        original_slots = encoded_outside_slots(
            dataset, objects, calib, standard_transform
        )
        standard_outside += len(original_slots)
        standard_images += bool(original_slots)

        for pass_index in range(passes):
            seed = augmentation_seed(
                base_seed, pass_index, item, dataset_size
            )
            center, crop_size = sample_crop_parameters(
                seed, image_size, scale, shift
            )
            transform = get_affine_transform(
                center, crop_size, 0, dataset.resolution
            )
            slots = encoded_outside_slots(
                dataset, objects, calib, transform
            )
            if slots:
                selected.append({
                    "item": item,
                    "image_id": image_id,
                    "pass_index": pass_index,
                    "seed": seed,
                    "outside_slots": slots,
                })
    return selected, standard_outside, standard_images


class SeededAugmentedSubset(Dataset):
    def __init__(self, dataset, sample_specs):
        self.dataset = dataset
        self.sample_specs = sample_specs

    def __len__(self):
        return len(self.sample_specs)

    def __getitem__(self, index):
        spec = self.sample_specs[index]
        state = np.random.get_state()
        np.random.seed(spec["seed"])
        try:
            inputs, calib, targets, info = self.dataset[spec["item"]]
        finally:
            np.random.set_state(state)
        info = dict(info)
        info["source_item"] = spec["item"]
        info["augmentation_seed"] = spec["seed"]
        info["pass_index"] = spec["pass_index"]
        return inputs, calib, targets, info


def make_record(
        batch_index, target_index, slot, targets, info, outside):
    return {
        "image_id": int(info["img_id"][batch_index]),
        "source_item": int(info["source_item"][batch_index]),
        "pass_index": int(info["pass_index"][batch_index]),
        "augmentation_seed": int(info["augmentation_seed"][batch_index]),
        "target_slot": int(slot),
        "difficulty": DIFFICULTY_NAMES.get(
            int(targets["object_difficulty"][batch_index, slot]), "Unknown"
        ),
        "outside": bool(outside),
        "assigned": False,
        "query_id": -1,
        "score": float("nan"),
        "bbox_iou": float("nan"),
        "target_index": int(target_index),
    }


def analyze_batch(outputs, targets, info, score_threshold):
    pred_scores = outputs["pred_logits"].sigmoid()[:, :, 1]
    pred_boxes = box_ops.box_cxcylrtb_to_xyxy(outputs["pred_boxes"])
    records = []

    for batch_index in range(pred_boxes.shape[0]):
        valid_slots = torch.nonzero(
            targets["mask_2d"][batch_index], as_tuple=False
        ).squeeze(1)
        if len(valid_slots) == 0:
            continue
        target_boxes = box_ops.box_cxcywh_to_xyxy(
            targets["boxes"][batch_index, valid_slots]
        ).cpu().numpy()
        boxes = pred_boxes[batch_index].detach().cpu().numpy()
        scores = pred_scores[batch_index].detach().cpu().numpy()
        candidates = np.flatnonzero(scores >= score_threshold)

        target_records = []
        for target_index, slot_tensor in enumerate(valid_slots):
            slot = int(slot_tensor)
            outside = bool(
                targets["outside_center_mask"][batch_index, slot]
            )
            target_records.append(
                make_record(
                    batch_index,
                    target_index,
                    slot,
                    targets,
                    info,
                    outside,
                )
            )

        if len(candidates):
            overlaps = pairwise_iou(boxes[candidates], target_boxes)
            pred_rows, target_cols = linear_sum_assignment(-overlaps)
            for pred_row, target_index in zip(pred_rows, target_cols):
                query_id = int(candidates[pred_row])
                slot = int(valid_slots[target_index])
                record = target_records[target_index]
                record.update({
                    "assigned": True,
                    "query_id": query_id,
                    "score": float(scores[query_id]),
                    "bbox_iou": float(overlaps[pred_row, target_index]),
                })
                metrics = center_error_metrics(
                    outputs["pred_boxes"][
                        batch_index, query_id, :2
                    ].detach().cpu().numpy(),
                    outputs["pred_outside_center_offset"][
                        batch_index, query_id
                    ].detach().cpu().numpy(),
                    targets["projected_3d_center"][
                        batch_index, slot
                    ].cpu().numpy(),
                    targets["outside_center_offset"][
                        batch_index, slot
                    ].cpu().numpy(),
                )
                record.update(metrics)
        records.extend(target_records)
    return records


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def write_records(records, output_path):
    fieldnames = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(output_path, "w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def print_summary(name, summary):
    print("\n{}:".format(name))
    print(
        "  targets={} assigned={} matched@0.3/0.5/0.7={}/{}/{}".format(
            summary["targets"],
            summary["assigned"],
            summary["matched_iou_0.3"],
            summary["matched_iou_0.5"],
            summary["matched_iou_0.7"],
        )
    )
    if not summary.get("metric_matches"):
        return
    print(
        "  center error px, representative -> full: "
        "{:.3f} -> {:.3f} (delta {:+.3f})".format(
            summary["representative_error_px_mean"],
            summary["full_error_px_mean"],
            summary["error_delta_px_mean"],
        )
    )
    print(
        "  improved: {}/{} ({:.2%})".format(
            summary["improved"],
            summary["metric_matches"],
            summary["improved_fraction"],
        )
    )
    print(
        "  target/predicted offset norm px: {:.3f} / {:.3f}".format(
            summary["target_offset_norm_px_mean"],
            summary["predicted_offset_norm_px_mean"],
        )
    )
    print(
        "  offset error px: {:.3f}; direction cosine: {}".format(
            summary["offset_error_px_mean"],
            (
                "{:.4f}".format(summary["offset_cosine_mean"])
                if summary["offset_cosine_mean"] is not None
                else "n/a"
            ),
        )
    )


def main():
    args = parse_args()
    with open(args.config, "r") as config_file:
        config = yaml.load(config_file, Loader=yaml.Loader)
    dataset_cfg = dict(config["dataset"])
    dataset_cfg["writelist"] = ["Car"]
    dataset_cfg["use_outside_center_modeling"] = True
    scale = dataset_cfg.get("scale", 0.05) if args.scale is None else args.scale
    shift = dataset_cfg.get("shift", 0.05) if args.shift is None else args.shift

    dataset = KITTI_Dataset(split=args.split, cfg=dataset_cfg)
    dataset.data_augmentation = True
    dataset.aug_pd = False
    dataset.aug_crop = True
    dataset.random_flip = 0.0
    dataset.random_crop = 1.0
    dataset.scale = scale
    dataset.shift = shift

    sample_specs, standard_outside, standard_images = find_augmented_samples(
        dataset, args.passes, args.seed, scale, shift
    )
    if not sample_specs:
        raise RuntimeError("No deterministic crop produced an outside-center target.")

    expected_outside = sum(
        len(spec["outside_slots"]) for spec in sample_specs
    )
    print("\nstandard {} outside targets: {} in {} images".format(
        args.split, standard_outside, standard_images
    ))
    print(
        "selected augmented samples: {} with {} outside target instances".format(
            len(sample_specs), expected_outside
        )
    )

    subset = SeededAugmentedSubset(dataset, sample_specs)
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
    )

    model, _ = build_model(config["model"], dataset_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    checkpoint_path = (
        args.checkpoint
        or config["tester"].get("checkpoint_path")
        or os.path.join(
            config["trainer"]["save_path"],
            config["model_name"],
            "checkpoint_best.pth",
        )
    )
    logger = logging.getLogger("outside_center_compensation")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    epoch, _, _ = load_checkpoint(
        model=model,
        optimizer=None,
        filename=checkpoint_path,
        map_location=device,
        logger=logger,
    )

    model.eval()
    records = []
    with torch.no_grad():
        for inputs, calibs, targets, info in tqdm(
                dataloader, desc="Evaluating selected augmented samples"):
            inputs = inputs.to(device)
            calibs = calibs.to(device)
            img_sizes = info["img_size"].to(device)
            outputs = model(
                inputs, calibs, targets, img_sizes, dn_args=0
            )
            records.extend(
                analyze_batch(
                    outputs,
                    targets,
                    info,
                    args.score_threshold,
                )
            )

    outside_records = [record for record in records if record["outside"]]
    inside_records = [record for record in records if not record["outside"]]
    outside_summary = summarize_records(outside_records, args.match_iou)
    inside_summary = summarize_records(inside_records, args.match_iou)
    outside_at_iou = {
        str(threshold): summarize_records(outside_records, threshold)
        for threshold in (0.3, 0.5, 0.7)
    }
    outside_unique_objects = summarize_unique_source_objects(
        outside_records, args.match_iou
    )
    by_difficulty = {
        name: summarize_records(
            [
                record
                for record in outside_records
                if record["difficulty"] == name
            ],
            args.match_iou,
        )
        for name in ("Easy", "Moderate", "Hard")
    }

    summary = {
        "config": args.config,
        "checkpoint": checkpoint_path,
        "checkpoint_epoch": epoch,
        "split": args.split,
        "passes": args.passes,
        "seed": args.seed,
        "scale": scale,
        "shift": shift,
        "score_threshold": args.score_threshold,
        "match_iou": args.match_iou,
        "standard_outside_targets": standard_outside,
        "standard_outside_images": standard_images,
        "selected_augmented_samples": len(sample_specs),
        "expected_outside_target_instances": expected_outside,
        "outside": outside_summary,
        "outside_at_iou": outside_at_iou,
        "outside_unique_source_objects": outside_unique_objects,
        "inside_same_images": inside_summary,
        "outside_by_difficulty": by_difficulty,
        "outside_difficulty_counts": dict(
            Counter(record["difficulty"] for record in outside_records)
        ),
        "unique_outside_source_objects": len({
            (record["image_id"], record["target_slot"])
            for record in outside_records
        }),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    write_records(
        records, os.path.join(args.output_dir, "matched_targets.csv")
    )
    with open(
            os.path.join(args.output_dir, "summary.json"), "w") as output_file:
        json.dump(json_ready(summary), output_file, indent=2, sort_keys=True)

    print("\ncheckpoint epoch: {}".format(epoch))
    print_summary("outside targets", outside_summary)
    print_summary(
        "outside unique source objects", outside_unique_objects
    )
    print("\noutside compensation across matching thresholds:")
    for threshold in (0.3, 0.5, 0.7):
        threshold_summary = outside_at_iou[str(threshold)]
        print(
            "  IoU>={:.1f}: n={} delta={:+.3f}px improved={:.2%}".format(
                threshold,
                threshold_summary.get("metric_matches", 0),
                threshold_summary.get("error_delta_px_mean", float("nan")),
                threshold_summary.get("improved_fraction", float("nan")),
            )
        )
    print_summary("inside targets in the same augmented images", inside_summary)
    for name in ("Easy", "Moderate", "Hard"):
        print_summary("outside {}".format(name), by_difficulty[name])
    print("\nresults: {}".format(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()
