import logging
import os
import tempfile
import unittest
from unittest import mock

import torch
import yaml
from torch import nn

from lib.helpers.decode_helper import extract_dedicated_dets_from_outputs
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.save_helper import load_checkpoint
from lib.models.monodetr.matcher import HungarianMatcher
from lib.models.monodetr.monodetr import SetCriterion
from lib.models.monodetr.monodetr import split_targets_by_outside_mask
import lib.models.monodetr.monodetr as monodetr_module
from tests.test_outside_center_head import build_dummy_model
from tests.test_outside_center_head import forward_inputs


class ZeroDDNLoss(nn.Module):
    def forward(self, depth_logits, *args, **kwargs):
        return depth_logits.sum() * 0.0


def make_target(outside_flags):
    count = len(outside_flags)
    outside_mask = torch.tensor(outside_flags, dtype=torch.bool)
    projected_centers = torch.tensor(
        [[0.4 + 0.05 * index, 0.5] for index in range(count)],
        dtype=torch.float32,
    )
    representative_centers = projected_centers.clone()
    representative_centers[outside_mask, 0] = 0.02
    offsets = projected_centers - representative_centers
    margins = torch.full((count, 4), 0.1)
    boxes_3d = torch.cat([projected_centers, margins], dim=1)
    representative_boxes = torch.cat(
        [representative_centers, margins], dim=1
    )
    return {
        "labels": torch.ones(count, dtype=torch.int64),
        "boxes": torch.tensor(
            [[0.5, 0.5, 0.2, 0.2] for _ in range(count)],
            dtype=torch.float32,
        ),
        "boxes_3d": boxes_3d,
        "boxes_3d_representative": representative_boxes,
        "representative_box_valid_mask": torch.ones(
            count, dtype=torch.bool
        ),
        "outside_center_offset": offsets,
        "outside_center_mask": outside_mask,
        "projected_3d_center": projected_centers,
        "representative_center": representative_centers,
        "depth": torch.full((count, 1), 20.0),
        "size_3d": torch.full((count, 3), 2.0),
        "heading_bin": torch.zeros(count, 1, dtype=torch.int64),
        "heading_res": torch.zeros(count, 1),
        "calibs": torch.zeros(count, 3, 4),
        "original_target_indices": torch.arange(count),
        "image_size": torch.tensor([1280.0, 384.0]),
    }


def build_dedicated_criterion():
    matcher = HungarianMatcher(
        cost_class=2.0,
        cost_3dcenter=10.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        use_outside_center_modeling=True,
    )
    with mock.patch.object(monodetr_module, "DDNLoss", ZeroDDNLoss):
        criterion = SetCriterion(
            num_classes=3,
            matcher=matcher,
            weight_dict={},
            focal_alpha=0.25,
            losses=[
                "labels", "boxes", "cardinality", "depths", "dims",
                "angles", "center", "depth_map",
            ],
            group_num=1,
            use_outside_center_modeling=True,
            use_dedicated_outside_queries=True,
        )
    criterion.ddn_loss = ZeroDDNLoss()
    return criterion


class DedicatedOutsideQueriesTest(unittest.TestCase):
    def build_model(self):
        model, transformer = build_dummy_model(
            True,
            use_dedicated_outside_queries=True,
            num_queries=50,
            num_outside_queries=10,
        )
        return model, transformer

    def test_output_shapes_and_auxiliary_groups(self):
        model, _ = self.build_model()
        model.eval()
        outputs = model(*forward_inputs())
        self.assertEqual(tuple(outputs["pred_logits"].shape), (2, 50, 3))
        self.assertEqual(tuple(outputs["pred_boxes"].shape), (2, 50, 6))
        self.assertEqual(
            tuple(outputs["pred_outside_logits"].shape), (2, 10, 3)
        )
        self.assertEqual(
            tuple(outputs["pred_outside_boxes"].shape), (2, 10, 6)
        )
        self.assertEqual(
            tuple(outputs["pred_outside_center_offset"].shape), (2, 10, 2)
        )
        self.assertNotIn("pred_outside_center_offset", outputs["aux_outputs"][0])
        self.assertEqual(len(outputs["aux_outputs_outside"]), 2)
        for auxiliary in outputs["aux_outputs_outside"]:
            self.assertEqual(
                tuple(auxiliary["pred_outside_logits"].shape), (2, 10, 3)
            )
            self.assertEqual(
                tuple(auxiliary["pred_outside_boxes"].shape), (2, 10, 6)
            )

    def test_query_embeddings_and_decoder_outputs_are_bidirectionally_isolated(self):
        torch.manual_seed(8)
        model, _ = self.build_model()
        model.eval()
        inputs = forward_inputs()
        before = model(*inputs)
        normal_before = before["pred_logits"].detach().clone()
        outside_before = before["pred_outside_logits"].detach().clone()

        with torch.no_grad():
            model.outside_query_embed.weight.add_(100.0)
        after_outside_change = model(*inputs)
        normal_difference = (
            after_outside_change["pred_logits"] - normal_before
        ).abs().max()
        outside_difference = (
            after_outside_change["pred_outside_logits"] - outside_before
        ).abs().max()
        self.assertLess(float(normal_difference), 1e-6)
        self.assertGreater(float(outside_difference), 1e-3)

        outside_after = (
            after_outside_change["pred_outside_logits"].detach().clone()
        )
        with torch.no_grad():
            model.query_embed.weight[:50].sub_(100.0)
        after_normal_change = model(*inputs)
        outside_difference = (
            after_normal_change["pred_outside_logits"] - outside_after
        ).abs().max()
        self.assertLess(float(outside_difference), 1e-6)

    def test_target_split_is_disjoint_complete_and_field_aligned(self):
        target = make_target([False, True, False, True])
        inside, outside = split_targets_by_outside_mask([target])
        inside_indices = inside[0]["original_target_indices"]
        outside_indices = outside[0]["original_target_indices"]
        self.assertEqual(inside_indices.tolist(), [0, 2])
        self.assertEqual(outside_indices.tolist(), [1, 3])
        self.assertEqual(
            set(inside_indices.tolist()) & set(outside_indices.tolist()),
            set(),
        )
        self.assertEqual(
            set(inside_indices.tolist()) | set(outside_indices.tolist()),
            {0, 1, 2, 3},
        )
        for key in (
                "labels", "boxes", "boxes_3d", "depth", "size_3d",
                "heading_bin", "outside_center_offset"):
            self.assertEqual(inside[0][key].shape[0], 2)
            self.assertEqual(outside[0][key].shape[0], 2)
        torch.testing.assert_close(
            inside[0]["image_size"], target["image_size"]
        )

    def test_dual_matcher_and_aux_losses_use_local_disjoint_indices(self):
        model, _ = self.build_model()
        model.train()
        outputs = model(*forward_inputs())
        criterion = build_dedicated_criterion()
        criterion.train()
        targets = [make_target([False, True]), make_target([True, False])]
        losses = criterion(outputs, targets)
        stats = criterion.last_match_stats
        self.assertEqual(stats["num_inside_gt"], 2)
        self.assertEqual(stats["num_outside_gt"], 2)
        for source, _ in stats["normal_indices"]:
            self.assertTrue((source >= 0).all() and (source < 50).all())
        for source, _ in stats["outside_indices"]:
            self.assertTrue((source >= 0).all() and (source < 10).all())
        for normal, outside in zip(
                stats["normal_original_target_indices"],
                stats["outside_original_target_indices"]):
            self.assertFalse(
                set(normal.tolist()) & set(outside.tolist())
            )
        expected = {
            "loss_ce",
            "loss_outside_ce",
            "loss_outside_center_offset",
            "loss_ce_0",
            "loss_outside_ce_0",
            "loss_outside_center_offset_0",
        }
        self.assertTrue(expected.issubset(losses))
        self.assertNotIn("loss_inside_center_offset_zero", losses)
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))

    def test_normal_and_outside_losses_have_isolated_query_gradients(self):
        targets = [make_target([False, True]), make_target([True, False])]

        model, _ = self.build_model()
        model.train()
        criterion = build_dedicated_criterion()
        losses = criterion(model(*forward_inputs()), targets)
        normal_total = sum(
            value for key, value in losses.items()
            if (
                key.startswith("loss_")
                and not key.startswith("loss_outside_")
                and key != "loss_depth_map"
            )
        )
        normal_total.backward()
        self.assertGreater(
            float(model.query_embed.weight.grad[:50].abs().sum()), 0.0
        )
        self.assertIsNone(model.outside_query_embed.weight.grad)

        model, _ = self.build_model()
        model.train()
        criterion = build_dedicated_criterion()
        losses = criterion(model(*forward_inputs()), targets)
        outside_total = sum(
            value for key, value in losses.items()
            if key.startswith("loss_outside_")
        )
        outside_total.backward()
        self.assertGreater(
            float(model.outside_query_embed.weight.grad.abs().sum()), 0.0
        )
        self.assertIsNone(model.query_embed.weight.grad)
        for head in model.outside_center_offset_embed:
            self.assertIsNotNone(head.layers[-1].bias.grad)

    def test_empty_outside_and_empty_inside_keep_background_classification(self):
        model, _ = self.build_model()
        model.train()
        criterion = build_dedicated_criterion()
        losses = criterion(
            model(*forward_inputs()),
            [make_target([False]), make_target([False])],
        )
        self.assertTrue(torch.isfinite(losses["loss_outside_ce"]))
        self.assertGreater(float(losses["loss_outside_ce"]), 0.0)
        for key in (
                "loss_outside_bbox", "loss_outside_giou",
                "loss_outside_depth", "loss_outside_dim",
                "loss_outside_angle", "loss_outside_center",
                "loss_outside_center_offset"):
            self.assertEqual(float(losses[key]), 0.0)
            self.assertTrue(losses[key].requires_grad)

        losses = criterion(
            model(*forward_inputs()),
            [make_target([True]), make_target([True])],
        )
        self.assertTrue(torch.isfinite(losses["loss_ce"]))
        self.assertGreater(float(losses["loss_ce"]), 0.0)
        for key in (
                "loss_bbox", "loss_giou", "loss_depth", "loss_dim",
                "loss_angle", "loss_center"):
            self.assertEqual(float(losses[key]), 0.0)
            self.assertTrue(losses[key].requires_grad)

    def test_checkpoint_allows_only_new_outside_query_embedding(self):
        source, _ = build_dummy_model(
            True,
            use_dedicated_outside_queries=False,
            num_queries=50,
        )
        target, _ = self.build_model()
        checkpoint = {
            "epoch": 159,
            "model_state": source.state_dict(),
            "optimizer_state": None,
            "best_result": 0.0,
            "best_epoch": 0,
        }
        logger = logging.getLogger("dedicated-checkpoint-test")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            torch.save(checkpoint, path)
            epoch, _, _ = load_checkpoint(
                target, None, path, map_location="cpu", logger=logger
            )
        self.assertEqual(epoch, 159)
        self.assertEqual(
            tuple(target.outside_query_embed.weight.shape), (10, 64)
        )

    def test_optimizer_groups_are_disjoint_and_use_configured_lrs(self):
        model, _ = self.build_model()
        optimizer = build_optimizer({
            "type": "adamw",
            "lr": 2e-5,
            "outside_query_lr": 1e-4,
            "outside_offset_head_lr": 5e-5,
            "weight_decay": 1e-4,
        }, model)
        groups = {
            group["group_name"]: group for group in optimizer.param_groups
        }
        self.assertEqual(groups["base_weight"]["lr"], 2e-5)
        self.assertEqual(groups["outside_query"]["lr"], 1e-4)
        self.assertEqual(groups["outside_offset_weight"]["lr"], 5e-5)
        parameter_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))

    def test_grouped_topk_keeps_low_scoring_outside_queries(self):
        batch = 1
        outputs = {
            "pred_logits": torch.full((batch, 50, 3), 10.0),
            "pred_boxes": torch.full((batch, 50, 6), 0.5),
            "pred_3d_dim": torch.ones(batch, 50, 3),
            "pred_depth": torch.ones(batch, 50, 2),
            "pred_angle": torch.zeros(batch, 50, 24),
            "pred_outside_logits": torch.full((batch, 10, 3), -10.0),
            "pred_outside_boxes": torch.full((batch, 10, 6), 0.5),
            "pred_outside_3d_dim": torch.ones(batch, 10, 3),
            "pred_outside_depth": torch.ones(batch, 10, 2),
            "pred_outside_angle": torch.zeros(batch, 10, 24),
            "pred_outside_center_offset": torch.zeros(batch, 10, 2),
        }
        normal, outside, debug = extract_dedicated_dets_from_outputs(
            outputs,
            normal_topk=50,
            outside_topk=10,
            return_debug=True,
        )
        self.assertEqual(tuple(normal.shape), (1, 50, 37))
        self.assertEqual(tuple(outside.shape), (1, 10, 37))
        self.assertTrue((debug["outside"]["scores"] < 0.001).all())
        self.assertTrue((debug["outside"]["query_ids"] < 10).all())

    def test_dedicated_mode_adds_only_the_independent_query_embedding(self):
        unified, _ = build_dummy_model(
            True,
            use_dedicated_outside_queries=False,
            num_queries=50,
        )
        dedicated, _ = self.build_model()
        unified_count = sum(parameter.numel() for parameter in unified.parameters())
        dedicated_count = sum(
            parameter.numel() for parameter in dedicated.parameters()
        )
        self.assertEqual(
            dedicated_count - unified_count,
            dedicated.outside_query_embed.weight.numel(),
        )

    def test_training_and_smoke_configs_keep_fifty_plus_ten_queries(self):
        for path in (
                "configs/monodetr.yaml",
                "configs/monodetr_dedicated_smoke.yaml"):
            with open(path, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.Loader)
            self.assertTrue(
                config["model"]["use_dedicated_outside_queries"]
            )
            self.assertEqual(config["model"]["num_queries"], 50)
            self.assertEqual(config["model"]["num_outside_queries"], 10)
            self.assertEqual(config["tester"]["normal_topk"], 50)
            self.assertEqual(config["tester"]["outside_topk"], 10)
            self.assertEqual(
                config["optimizer"]["outside_query_lr"], 1e-4
            )


if __name__ == "__main__":
    unittest.main()
