import importlib
import math
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from lib.datasets.kitti.outside_center_utils import (
    build_representative_box_target,
)
from lib.models.monodetr.matcher import HungarianMatcher
from tests.test_outside_center_head import (
    DummyBackbone,
    DummyDepthPredictor,
    DummyTransformer,
    build_dummy_model,
    forward_inputs,
)


monodetr_module = importlib.import_module("lib.models.monodetr.monodetr")


class DummyDDNLoss(nn.Module):
    pass


def build_criterion(matcher, losses, enabled=True):
    weight_dict = {
        "loss_ce": 2.0,
        "loss_bbox": 5.0,
        "loss_giou": 2.0,
        "loss_center": 10.0,
    }
    if enabled:
        weight_dict["loss_outside_center_offset"] = 1.0
        weight_dict["loss_outside_center_offset_0"] = 1.0
        weight_dict["loss_outside_center_offset_1"] = 1.0
        weight_dict["loss_inside_center_offset_zero"] = 0.1
        weight_dict["loss_inside_center_offset_zero_0"] = 0.1
        weight_dict["loss_inside_center_offset_zero_1"] = 0.1
    with mock.patch.object(monodetr_module, "DDNLoss", DummyDDNLoss):
        return monodetr_module.SetCriterion(
            num_classes=3,
            matcher=matcher,
            weight_dict=weight_dict,
            focal_alpha=0.25,
            losses=losses,
            group_num=1,
            use_outside_center_modeling=enabled,
        )


def representative_target(
    center,
    margins,
    projected_center=None,
    offset=(0.0, 0.0),
    outside=False,
    valid=True,
    label=1,
):
    center = torch.tensor(center, dtype=torch.float32)
    margins = torch.tensor(margins, dtype=torch.float32)
    representative_box = torch.cat([center, margins]).unsqueeze(0)
    if projected_center is None:
        projected_center = center
    projected_center = torch.as_tensor(
        projected_center, dtype=torch.float32
    ).reshape(1, 2)
    return {
        "labels": torch.tensor([label], dtype=torch.int64),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], dtype=torch.float32),
        "boxes_3d": torch.cat(
            [projected_center, margins.unsqueeze(0)], dim=1
        ),
        "boxes_3d_representative": representative_box,
        "representative_box_valid_mask": torch.tensor([valid]),
        "outside_center_offset": torch.tensor(
            [offset], dtype=torch.float32
        ),
        "outside_center_mask": torch.tensor([outside]),
    }


class OutsideCenterTrainingTest(unittest.TestCase):
    def test_build_adds_offset_weights_only_when_enabled(self):
        base_cfg = {
            "num_classes": 3,
            "num_queries": 5,
            "aux_loss": True,
            "num_feature_levels": 3,
            "with_box_refine": True,
            "two_stage": False,
            "init_box": False,
            "use_dab": False,
            "two_stage_dino": False,
            "set_cost_class": 2.0,
            "set_cost_bbox": 5.0,
            "set_cost_3dcenter": 10.0,
            "set_cost_giou": 2.0,
            "cls_loss_coef": 2.0,
            "bbox_loss_coef": 5.0,
            "giou_loss_coef": 2.0,
            "dim_loss_coef": 1.0,
            "angle_loss_coef": 1.0,
            "depth_loss_coef": 1.0,
            "3dcenter_loss_coef": 10.0,
            "depth_map_loss_coef": 1.0,
            "outside_center_offset_loss_coef": 1.0,
            "inside_center_offset_zero_loss_coef": 0.1,
            "use_dn": False,
            "dec_layers": 3,
            "focal_alpha": 0.25,
            "device": "cpu",
        }

        def build_for(enabled):
            cfg = dict(base_cfg)
            cfg["use_outside_center_modeling"] = enabled
            with mock.patch.object(
                    monodetr_module, "build_backbone",
                    return_value=DummyBackbone(32)), mock.patch.object(
                    monodetr_module, "build_depthaware_transformer",
                    return_value=DummyTransformer(32, 3)), mock.patch.object(
                    monodetr_module, "DepthPredictor",
                    return_value=DummyDepthPredictor()), mock.patch.object(
                    monodetr_module, "DDNLoss", DummyDDNLoss):
                return monodetr_module.build(cfg)

        _, disabled_criterion = build_for(False)
        _, enabled_criterion = build_for(True)
        self.assertNotIn(
            "loss_outside_center_offset", disabled_criterion.weight_dict
        )
        self.assertNotIn(
            "outside_center_offset", disabled_criterion.losses
        )
        self.assertEqual(
            enabled_criterion.weight_dict["loss_outside_center_offset"], 1.0
        )
        self.assertEqual(
            enabled_criterion.weight_dict["loss_outside_center_offset_0"], 1.0
        )
        self.assertEqual(
            enabled_criterion.weight_dict["loss_outside_center_offset_1"], 1.0
        )
        self.assertNotIn(
            "loss_outside_center_offset_enc",
            enabled_criterion.weight_dict,
        )
        self.assertEqual(
            enabled_criterion.weight_dict[
                "loss_inside_center_offset_zero"
            ],
            0.1,
        )
        self.assertNotIn(
            "loss_inside_center_offset_zero_enc",
            enabled_criterion.weight_dict,
        )

    def test_representative_box_reconstructs_original_box(self):
        target, valid = build_representative_box_target(
            representative_center=[0.0, 25.0],
            bbox_xyxy=[0.0, 10.0, 40.0, 50.0],
            image_width=100,
            image_height=60,
        )
        self.assertTrue(valid)
        reconstructed = np.array(
            [
                target[0] - target[2],
                target[1] - target[4],
                target[0] + target[3],
                target[1] + target[5],
            ]
        )
        expected = np.array([0.0, 10.0 / 60.0, 0.4, 50.0 / 60.0])
        np.testing.assert_allclose(reconstructed, expected, atol=1e-7)

    def test_significant_negative_margin_is_preserved_and_marked_invalid(self):
        target, valid = build_representative_box_target(
            representative_center=[0.0, 25.0],
            bbox_xyxy=[5.0, 10.0, 40.0, 50.0],
            image_width=100,
            image_height=60,
        )
        self.assertFalse(valid)
        self.assertLess(target[2], 0.0)

    def test_matcher_uses_representative_center_without_offset_cost(self):
        matcher = HungarianMatcher(
            cost_class=0.0,
            cost_3dcenter=1.0,
            cost_bbox=1.0,
            cost_giou=1.0,
            use_outside_center_modeling=True,
        )
        outputs = {
            "pred_logits": torch.zeros(1, 2, 3),
            "pred_boxes": torch.tensor(
                [[
                    [0.10, 0.50, 0.1, 0.1, 0.1, 0.1],
                    [0.00, 0.50, 0.1, 0.1, 0.1, 0.1],
                ]]
            ),
            "pred_outside_center_offset": torch.tensor(
                [[[100.0, 100.0], [0.0, 0.0]]]
            ),
        }
        targets = [
            representative_target(
                center=(0.10, 0.50),
                margins=(0.1, 0.1, 0.1, 0.1),
                projected_center=(-0.50, 0.50),
                offset=(-0.60, 0.0),
                outside=True,
                valid=False,
            )
        ]
        indices = matcher(outputs, targets, group_num=1)
        self.assertEqual(indices[0][0].tolist(), [0])
        self.assertEqual(indices[0][1].tolist(), [0])

    def test_log1p_offset_loss_and_inside_mask_filter(self):
        matcher = HungarianMatcher(
            use_outside_center_modeling=True
        )
        criterion = build_criterion(
            matcher, ["outside_center_offset"], enabled=True
        )
        outputs = {
            "pred_outside_center_offset": torch.tensor(
                [[[7.0, 7.0], [0.0, 0.0]]], requires_grad=True
            )
        }
        targets = [{
            "outside_center_offset": torch.tensor(
                [[9.0, 9.0], [0.2, -0.1]]
            ),
            "outside_center_mask": torch.tensor([False, True]),
        }]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        loss = criterion.loss_outside_center_offset(
            outputs, targets, indices, num_boxes=2
        )["loss_outside_center_offset"]
        expected = math.log1p(0.2) + math.log1p(0.1)
        self.assertAlmostEqual(float(loss), expected, places=7)
        loss.backward()
        self.assertEqual(
            outputs["pred_outside_center_offset"].grad[0, 0].abs().sum(),
            0.0,
        )
        self.assertGreater(
            outputs["pred_outside_center_offset"].grad[0, 1].abs().sum(),
            0.0,
        )

    def test_no_outside_target_returns_differentiable_zero(self):
        matcher = HungarianMatcher(
            use_outside_center_modeling=True
        )
        criterion = build_criterion(
            matcher, ["outside_center_offset"], enabled=True
        )
        prediction = torch.randn(1, 1, 2, requires_grad=True)
        outputs = {"pred_outside_center_offset": prediction}
        targets = [{
            "outside_center_offset": torch.tensor([[3.0, -2.0]]),
            "outside_center_mask": torch.tensor([False]),
        }]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = criterion.loss_outside_center_offset(
            outputs, targets, indices, num_boxes=1
        )["loss_outside_center_offset"]
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(loss.device, prediction.device)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_inside_zero_loss_is_separately_normalized_and_masked(self):
        matcher = HungarianMatcher(
            use_outside_center_modeling=True
        )
        criterion = build_criterion(
            matcher, ["inside_center_offset_zero"], enabled=True
        )
        prediction = torch.tensor(
            [[[0.2, -0.1], [8.0, 8.0]]], requires_grad=True
        )
        outputs = {"pred_outside_center_offset": prediction}
        targets = [{
            "outside_center_mask": torch.tensor([False, True]),
        }]
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        loss = criterion.loss_inside_center_offset_zero(
            outputs, targets, indices, num_boxes=2
        )["loss_inside_center_offset_zero"]
        expected = math.log1p(0.2) + math.log1p(0.1)
        self.assertAlmostEqual(float(loss), expected, places=7)
        loss.backward()
        self.assertGreater(float(prediction.grad[0, 0].abs().sum()), 0.0)
        self.assertEqual(float(prediction.grad[0, 1].abs().sum()), 0.0)

    def test_no_inside_target_returns_differentiable_zero(self):
        matcher = HungarianMatcher(
            use_outside_center_modeling=True
        )
        criterion = build_criterion(
            matcher, ["inside_center_offset_zero"], enabled=True
        )
        prediction = torch.randn(1, 1, 2, requires_grad=True)
        outputs = {"pred_outside_center_offset": prediction}
        targets = [{
            "outside_center_mask": torch.tensor([True]),
        }]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = criterion.loss_inside_center_offset_zero(
            outputs, targets, indices, num_boxes=1
        )["loss_inside_center_offset_zero"]
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(prediction.grad, torch.zeros_like(prediction)))

    def test_invalid_representative_box_has_no_box_or_giou_loss(self):
        matcher = HungarianMatcher(
            use_outside_center_modeling=True
        )
        criterion = build_criterion(matcher, ["boxes"], enabled=True)
        pred_boxes = torch.tensor(
            [[[0.2, 0.5, 0.1, 0.1, 0.1, 0.1]]],
            requires_grad=True,
        )
        outputs = {"pred_boxes": pred_boxes}
        targets = [
            representative_target(
                center=(0.0, 0.5),
                margins=(-0.1, 0.2, 0.1, 0.1),
                projected_center=(-0.2, 0.5),
                outside=True,
                valid=False,
            )
        ]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        losses = criterion.loss_boxes(outputs, targets, indices, num_boxes=1)
        self.assertEqual(float(losses["loss_bbox"]), 0.0)
        self.assertEqual(float(losses["loss_giou"]), 0.0)
        (losses["loss_bbox"] + losses["loss_giou"]).backward()
        self.assertTrue(torch.equal(pred_boxes.grad, torch.zeros_like(pred_boxes)))

    def test_single_batch_criterion_auxiliary_losses_and_backward(self):
        torch.manual_seed(7)
        model, transformer = build_dummy_model(True)
        model.train()
        outputs = model(*forward_inputs())
        matcher = HungarianMatcher(
            cost_class=2.0,
            cost_3dcenter=10.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            use_outside_center_modeling=True,
        )
        criterion = build_criterion(
            matcher,
            ["labels", "boxes", "cardinality", "center",
             "outside_center_offset", "inside_center_offset_zero"],
            enabled=True,
        )
        targets = [
            representative_target(
                center=(0.02, 0.50),
                margins=(0.02, 0.20, 0.10, 0.10),
                projected_center=(-0.03, 0.50),
                offset=(-0.05, 0.0),
                outside=True,
                valid=True,
                label=1,
            ),
            representative_target(
                center=(0.60, 0.50),
                margins=(0.10, 0.10, 0.10, 0.10),
                outside=False,
                valid=True,
                label=0,
            ),
        ]
        losses = criterion(outputs, targets)
        expected_offset_keys = {
            "loss_outside_center_offset",
            "loss_outside_center_offset_0",
            "loss_outside_center_offset_1",
            "loss_inside_center_offset_zero",
            "loss_inside_center_offset_zero_0",
            "loss_inside_center_offset_zero_1",
        }
        self.assertTrue(expected_offset_keys.issubset(losses))
        self.assertTrue(
            all(torch.isfinite(value).all() for value in losses.values())
        )
        self.assertGreater(float(losses["loss_outside_center_offset"]), 0.0)

        total_loss = sum(
            value * criterion.weight_dict[key]
            for key, value in losses.items()
            if key in criterion.weight_dict
        )
        total_loss.backward()
        for head in model.outside_center_offset_embed:
            self.assertGreater(
                float(head.layers[-1].bias.grad.abs().sum()), 0.0
            )
        self.assertGreater(
            float(model.bbox_embed[-1].layers[-1].weight.grad.abs().sum()),
            0.0,
        )
        self.assertIsNotNone(transformer.last_hs.grad)
        self.assertTrue(torch.isfinite(transformer.last_hs.grad).all())


if __name__ == "__main__":
    unittest.main()
