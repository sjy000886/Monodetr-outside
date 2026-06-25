import logging
import os
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn
import torch.nn.functional as F

from lib.helpers.model_helper import build_model
from lib.helpers.save_helper import load_checkpoint
from lib.models.monodetr.monodetr import MonoDETR
from utils.misc import NestedTensor


class DummyBackbone(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.strides = [4, 8, 16]
        self.num_channels = [hidden_dim, hidden_dim, hidden_dim]
        self.proj = nn.Conv2d(3, hidden_dim, kernel_size=1)

    def forward(self, images):
        feature = self.proj(images)
        features = []
        positions = []
        for size in (8, 4, 2):
            tensor = F.adaptive_avg_pool2d(feature, (size, size))
            mask = torch.zeros(
                tensor.shape[0], size, size, dtype=torch.bool, device=tensor.device
            )
            features.append(NestedTensor(tensor, mask))
            positions.append(torch.zeros_like(tensor))
        return features, positions


class DummyDecoder(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None


class DummyTransformer(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.d_model = hidden_dim
        self.decoder = DummyDecoder(num_layers)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.last_hs = None
        self.last_outside_hs = None

    def _decoder_outputs(self, srcs, query_embed):
        batch_size = srcs[0].shape[0]
        query_count = query_embed.shape[0]
        pooled = srcs[0].mean(dim=(2, 3))
        query_content = query_embed[:, self.d_model:].unsqueeze(0)
        hidden = (
            self.hidden_proj(pooled).unsqueeze(1)
            + query_content.expand(batch_size, -1, -1)
        )
        hs = torch.stack(
            [
                hidden + float(layer) / 10.0
                for layer in range(self.decoder.num_layers)
            ]
        )
        hs.retain_grad()
        references = torch.full(
            (batch_size, query_count, 2),
            0.5,
            dtype=hs.dtype,
            device=hs.device,
        )
        inter_references = references.unsqueeze(0).repeat(
            self.decoder.num_layers, 1, 1, 1
        )
        dimensions = torch.stack(
            [
                hidden[:, :, :3] + 2.0 + float(layer) / 10.0
                for layer in range(self.decoder.num_layers)
            ]
        )
        return hs, references, inter_references, dimensions

    def forward(
        self,
        srcs,
        masks,
        positions,
        query_embed,
        depth_pos_embed,
        depth_pos_embed_ip,
        outside_query_embed=None,
    ):
        hs, references, inter_references, dimensions = (
            self._decoder_outputs(srcs, query_embed)
        )
        self.last_hs = hs
        result = (
            hs, references, inter_references, dimensions, None, None
        )
        if outside_query_embed is not None:
            outside = self._decoder_outputs(srcs, outside_query_embed)
            self.last_outside_hs = outside[0]
            result = result + (outside,)
        return result


class DummyDepthPredictor(nn.Module):
    def forward(self, srcs, mask, position):
        depth_feature = srcs[1]
        weighted_depth = depth_feature.mean(dim=1)
        logits = depth_feature.new_zeros(
            depth_feature.shape[0], 81, depth_feature.shape[2], depth_feature.shape[3]
        )
        return logits, depth_feature, weighted_depth, depth_feature


def build_dummy_model(
        use_outside_center_modeling,
        use_dedicated_outside_queries=False,
        num_queries=5,
        num_outside_queries=10):
    hidden_dim = 32
    transformer = DummyTransformer(hidden_dim=hidden_dim, num_layers=3)
    model = MonoDETR(
        backbone=DummyBackbone(hidden_dim),
        depthaware_transformer=transformer,
        depth_predictor=DummyDepthPredictor(),
        num_classes=3,
        num_queries=num_queries,
        num_feature_levels=3,
        aux_loss=True,
        with_box_refine=True,
        two_stage=False,
        init_box=False,
        use_dab=False,
        group_num=1,
        two_stage_dino=False,
        use_outside_center_modeling=use_outside_center_modeling,
        use_dedicated_outside_queries=use_dedicated_outside_queries,
        num_outside_queries=num_outside_queries,
    )
    return model, transformer


def forward_inputs():
    images = torch.randn(2, 3, 32, 32)
    calibs = torch.zeros(2, 3, 4)
    calibs[:, 0, 0] = 700.0
    img_sizes = torch.tensor([[32.0, 32.0], [32.0, 32.0]])
    return images, calibs, None, img_sizes


class OutsideCenterHeadTest(unittest.TestCase):
    def test_disabled_output_and_parameter_surface_are_unchanged(self):
        model, _ = build_dummy_model(False)
        model.eval()
        outputs = model(*forward_inputs())

        self.assertFalse(hasattr(model, "outside_center_offset_embed"))
        self.assertNotIn("pred_outside_center_offset", outputs)
        self.assertEqual(
            set(outputs),
            {
                "pred_logits",
                "pred_boxes",
                "pred_3d_dim",
                "pred_depth",
                "pred_angle",
                "pred_depth_map_logits",
                "aux_outputs",
            },
        )
        for auxiliary in outputs["aux_outputs"]:
            self.assertNotIn("pred_outside_center_offset", auxiliary)
        self.assertFalse(
            any(
                name.startswith("outside_center_offset_embed.")
                for name, _ in model.named_parameters()
            )
        )

    def test_enabled_main_and_auxiliary_outputs_are_zero_initialized(self):
        model, _ = build_dummy_model(True)
        model.eval()
        outputs = model(*forward_inputs())

        offset = outputs["pred_outside_center_offset"]
        self.assertEqual(tuple(offset.shape), (2, 5, 2))
        self.assertEqual(offset.dtype, outputs["pred_boxes"].dtype)
        self.assertEqual(offset.device, outputs["pred_boxes"].device)
        self.assertTrue(torch.isfinite(offset).all())
        self.assertEqual(float(offset.abs().max()), 0.0)
        self.assertEqual(len(outputs["aux_outputs"]), 2)
        for auxiliary in outputs["aux_outputs"]:
            auxiliary_offset = auxiliary["pred_outside_center_offset"]
            self.assertEqual(tuple(auxiliary_offset.shape), (2, 5, 2))
            self.assertEqual(float(auxiliary_offset.abs().max()), 0.0)

        for head in model.outside_center_offset_embed:
            self.assertEqual(float(head.layers[-1].weight.abs().max()), 0.0)
            self.assertEqual(float(head.layers[-1].bias.abs().max()), 0.0)

    def test_output_is_unbounded_and_uses_corresponding_decoder_layer(self):
        model, _ = build_dummy_model(True)
        model.eval()
        with torch.no_grad():
            for layer, head in enumerate(model.outside_center_offset_embed):
                head.layers[-1].bias.copy_(
                    torch.tensor([-2.0 - layer, 2.0 + layer])
                )

        outputs = model(*forward_inputs())
        self.assertTrue(
            torch.allclose(
                outputs["aux_outputs"][0]["pred_outside_center_offset"][0, 0],
                torch.tensor([-2.0, 2.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["aux_outputs"][1]["pred_outside_center_offset"][0, 0],
                torch.tensor([-3.0, 3.0]),
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["pred_outside_center_offset"][0, 0],
                torch.tensor([-4.0, 4.0]),
            )
        )

    def test_dummy_backward_connects_head_and_decoder_hidden_state(self):
        model, transformer = build_dummy_model(True)
        model.train()
        outputs = model(*forward_inputs())
        outputs["pred_outside_center_offset"].sum().backward()

        for parameter in model.outside_center_offset_embed[-1].parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertIsNotNone(transformer.last_hs.grad)
        self.assertTrue(torch.isfinite(transformer.last_hs.grad).all())
        model.zero_grad()

    def test_model_helper_passes_the_dataset_switch(self):
        sentinel = object()
        with mock.patch(
            "lib.helpers.model_helper.build_monodetr", return_value=sentinel
        ) as mocked_build:
            result = build_model(
                {"num_classes": 3},
                {"use_outside_center_modeling": True},
            )
        self.assertIs(result, sentinel)
        self.assertTrue(
            mocked_build.call_args.args[0]["use_outside_center_modeling"]
        )

    def test_original_checkpoint_is_directionally_compatible(self):
        disabled_model, _ = build_dummy_model(False)
        enabled_model, _ = build_dummy_model(True)
        logger = logging.getLogger("outside-center-checkpoint-test")

        checkpoint = {
            "epoch": 1,
            "model_state": disabled_model.state_dict(),
            "optimizer_state": None,
            "best_result": 0.0,
            "best_epoch": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            torch.save(checkpoint, path)
            load_checkpoint(
                disabled_model, None, path, map_location="cpu", logger=logger
            )
            load_checkpoint(
                enabled_model, None, path, map_location="cpu", logger=logger
            )

        for head in enabled_model.outside_center_offset_embed:
            self.assertEqual(float(head.layers[-1].weight.abs().max()), 0.0)
            self.assertEqual(float(head.layers[-1].bias.abs().max()), 0.0)

    def test_checkpoint_compatibility_does_not_hide_original_missing_keys(self):
        disabled_model, _ = build_dummy_model(False)
        enabled_model, _ = build_dummy_model(True)
        logger = logging.getLogger("outside-center-invalid-checkpoint-test")
        state_dict = disabled_model.state_dict()
        state_dict.pop("class_embed.0.weight")
        checkpoint = {
            "epoch": 1,
            "model_state": state_dict,
            "optimizer_state": None,
            "best_result": 0.0,
            "best_epoch": 0,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(RuntimeError, "class_embed.0.weight"):
                load_checkpoint(
                    enabled_model,
                    None,
                    path,
                    map_location="cpu",
                    logger=logger,
                )

    def test_legacy_optimizer_state_is_migrated_for_the_new_head(self):
        disabled_model, _ = build_dummy_model(False)
        enabled_model, _ = build_dummy_model(True)
        disabled_optimizer = torch.optim.Adam(disabled_model.parameters(), lr=1e-4)
        enabled_optimizer = torch.optim.Adam(enabled_model.parameters(), lr=1e-4)
        logger = logging.getLogger("outside-center-optimizer-checkpoint-test")

        disabled_outputs = disabled_model(*forward_inputs())
        disabled_outputs["pred_boxes"].sum().backward()
        disabled_optimizer.step()
        checkpoint = {
            "epoch": 1,
            "model_state": disabled_model.state_dict(),
            "optimizer_state": disabled_optimizer.state_dict(),
            "best_result": 0.0,
            "best_epoch": 0,
        }

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pth")
            torch.save(checkpoint, path)
            load_checkpoint(
                enabled_model,
                enabled_optimizer,
                path,
                map_location="cpu",
                logger=logger,
            )

        offset_parameters = {
            parameter
            for head in enabled_model.outside_center_offset_embed
            for parameter in head.parameters()
        }
        self.assertTrue(enabled_optimizer.state)
        self.assertTrue(
            all(
                not enabled_optimizer.state.get(parameter)
                for parameter in offset_parameters
            )
        )


if __name__ == "__main__":
    unittest.main()
