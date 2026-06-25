import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
import torch
import yaml

from lib.datasets.kitti.kitti_utils import Calibration
from lib.helpers.decode_helper import (
    decode_detections,
    decode_projected_center,
    extract_dets_from_outputs,
    normalized_center_to_image,
    resolve_outside_center_decode_mode,
    select_projected_center,
    validate_outside_center_decode_mode,
)
from lib.helpers.tester_helper import Tester


def make_calibration():
    return Calibration({
        "P2": np.array(
            [
                [1000.0, 0.0, 640.0, 0.0],
                [0.0, 1000.0, 192.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        "R0": np.eye(3, dtype=np.float32),
        "Tr_velo2cam": np.zeros((3, 4), dtype=np.float32),
    })


def make_outputs(center=(0.4, 0.6), offset=None):
    logits = torch.full((1, 2, 3), -10.0)
    logits[0, 1, 2] = 10.0
    boxes = torch.tensor(
        [[
            [0.2, 0.3, 0.05, 0.05, 0.05, 0.05],
            [center[0], center[1], 0.1, 0.2, 0.1, 0.15],
        ]],
        dtype=torch.float32,
    )
    outputs = {
        "pred_logits": logits,
        "pred_boxes": boxes,
        "pred_angle": torch.zeros(1, 2, 24),
        "pred_3d_dim": torch.zeros(1, 2, 3),
        "pred_depth": torch.tensor(
            [[[15.0, 0.0], [20.0, 0.0]]]
        ),
    }
    if offset is not None:
        outputs["pred_outside_center_offset"] = torch.tensor(
            [[[99.0, 99.0], [offset[0], offset[1]]]],
            dtype=torch.float32,
        )
    return outputs


class OutsideCenterDecodeTest(unittest.TestCase):
    def test_unbounded_center_decode_directions(self):
        cases = (
            ([0.0, 0.5], [-0.2, 0.0], [-0.2, 0.5]),
            ([1.0, 0.5], [0.15, 0.0], [1.15, 0.5]),
            ([0.5, 0.0], [0.0, -0.1], [0.5, -0.1]),
            ([0.4, 0.6], [0.0, 0.0], [0.4, 0.6]),
        )
        for representative, offset, expected in cases:
            decoded = decode_projected_center(
                torch.tensor(representative),
                torch.tensor(offset),
            )
            torch.testing.assert_close(
                decoded, torch.tensor(expected), rtol=0, atol=0
            )

    def test_topk_gathers_matching_offset_and_keeps_2d_box(self):
        original = extract_dets_from_outputs(
            make_outputs(center=(1.0, 0.5)),
            topk=1,
            outside_center_decode_mode="legacy",
        )
        decoded = extract_dets_from_outputs(
            make_outputs(center=(1.0, 0.5), offset=(0.15, 0.0)),
            topk=1,
            outside_center_decode_mode="full",
        )
        torch.testing.assert_close(
            original[:, :, 0:34], decoded[:, :, 0:34], rtol=0, atol=0
        )
        torch.testing.assert_close(
            decoded[0, 0, 34:36],
            torch.tensor([1.15, 0.5]),
            rtol=0,
            atol=1e-7,
        )

    def test_zero_offset_detection_tensor_is_identical(self):
        original = extract_dets_from_outputs(
            make_outputs(),
            topk=1,
            outside_center_decode_mode="legacy",
        )
        enabled = extract_dets_from_outputs(
            make_outputs(offset=(0.0, 0.0)),
            topk=1,
            outside_center_decode_mode="zero_offset_new",
        )
        torch.testing.assert_close(original, enabled, rtol=0, atol=0)

    def test_camera_back_projection_uses_decoded_center(self):
        calibration = make_calibration()
        dets = extract_dets_from_outputs(
            make_outputs(center=(0.5, 0.5), offset=(-0.2, 0.0)),
            topk=1,
            outside_center_decode_mode="full",
        ).numpy()
        info = {
            "img_size": np.array([[1280.0, 384.0]], dtype=np.float32),
            "img_id": np.array([7]),
        }
        results = decode_detections(
            dets,
            info,
            [calibration],
            np.zeros((3, 3), dtype=np.float32),
            threshold=0.0,
            outside_center_decode_mode="full",
        )
        prediction = results[7][0]
        decoded_u = (0.5 - 0.2) * 1280.0
        decoded_v = 0.5 * 384.0
        expected_x = (decoded_u - 640.0) * 20.0 / 1000.0
        expected_y = (decoded_v - 192.0) * 20.0 / 1000.0
        self.assertAlmostEqual(prediction[9], expected_x, places=6)
        self.assertAlmostEqual(prediction[10], expected_y, places=6)
        self.assertAlmostEqual(prediction[11], 20.0, places=6)
        self.assertAlmostEqual(
            prediction[12],
            calibration.alpha2ry(0.0, decoded_u),
            places=6,
        )
        self.assertTrue(np.isfinite(prediction).all())

    def test_nonuniform_image_scaling_is_applied_once(self):
        center = np.array([-0.2, 1.1], dtype=np.float32)
        image_size = np.array([1242.0, 375.0], dtype=np.float32)
        decoded = normalized_center_to_image(center, image_size)
        np.testing.assert_allclose(
            decoded, [-248.4, 412.5], rtol=0, atol=1e-5
        )

    def test_zero_offset_kitti_decode_is_identical_when_centers_coincide(self):
        calibration = make_calibration()
        outputs = make_outputs(center=(0.5, 0.5), offset=(0.0, 0.0))
        outputs["pred_boxes"][0, 1, 2:6] = torch.tensor(
            [0.1, 0.1, 0.1, 0.1]
        )
        dets = extract_dets_from_outputs(
            outputs,
            topk=1,
            outside_center_decode_mode="zero_offset_new",
        ).numpy()
        info = {
            "img_size": np.array([[1280.0, 384.0]], dtype=np.float32),
            "img_id": np.array([3]),
        }
        kwargs = dict(
            dets=dets,
            info=info,
            calibs=[calibration],
            cls_mean_size=np.zeros((3, 3), dtype=np.float32),
            threshold=0.0,
        )
        legacy = decode_detections(
            outside_center_decode_mode="legacy", **kwargs
        )
        enabled = decode_detections(
            outside_center_decode_mode="zero_offset_new", **kwargs
        )
        np.testing.assert_allclose(legacy[3], enabled[3], rtol=0, atol=0)

    def test_kitti_writer_keeps_original_field_count(self):
        tester = object.__new__(Tester)
        tester.dataset_type = "KITTI"
        tester.class_name = ["Pedestrian", "Car", "Cyclist"]
        with tempfile.TemporaryDirectory() as directory:
            tester.output_dir = directory
            tester.save_results({
                7: [[
                    1, 0.1,
                    10.0, 20.0, 30.0, 40.0,
                    1.5, 1.6, 3.8,
                    2.0, 1.0, 20.0,
                    0.2, 0.9,
                ]]
            })
            with open(
                    directory + "/outputs/data/000007.txt", "r") as result_file:
                fields = result_file.readline().split()
        self.assertEqual(len(fields), 16)
        self.assertEqual(fields[0], "Car")
        self.assertTrue(all(np.isfinite(float(value)) for value in fields[1:]))

    def test_disabled_extract_matches_original_implementation(self):
        source = subprocess.check_output(
            ["git", "show", "HEAD:lib/helpers/decode_helper.py"],
            text=True,
        )
        module_name = "lib.helpers.original_decode_stage4"
        original_module = types.ModuleType(module_name)
        original_module.__package__ = "lib.helpers"
        sys.modules[module_name] = original_module
        exec(
            compile(source, "original_decode_stage4.py", "exec"),
            original_module.__dict__,
        )

        outputs = make_outputs()
        original = original_module.extract_dets_from_outputs(
            outputs, topk=1
        )
        current = extract_dets_from_outputs(
            outputs,
            topk=1,
            outside_center_decode_mode="legacy",
        )
        torch.testing.assert_close(original, current, rtol=0, atol=0)

    def test_disabled_kitti_decode_matches_original_implementation(self):
        source = subprocess.check_output(
            ["git", "show", "HEAD:lib/helpers/decode_helper.py"],
            text=True,
        )
        module_name = "lib.helpers.original_kitti_decode_stage4"
        original_module = types.ModuleType(module_name)
        original_module.__package__ = "lib.helpers"
        sys.modules[module_name] = original_module
        exec(
            compile(source, "original_kitti_decode_stage4.py", "exec"),
            original_module.__dict__,
        )

        dets = extract_dets_from_outputs(
            make_outputs(),
            topk=1,
            outside_center_decode_mode="legacy",
        ).numpy()
        info = {
            "img_size": np.array([[1242.0, 375.0]], dtype=np.float32),
            "img_id": np.array([11]),
        }
        kwargs = dict(
            info=info,
            calibs=[make_calibration()],
            cls_mean_size=np.array(
                [
                    [1.7, 0.6, 0.8],
                    [1.5, 1.6, 3.8],
                    [1.7, 0.6, 1.7],
                ],
                dtype=np.float32,
            ),
            threshold=0.0,
        )
        original = original_module.decode_detections(
            dets=dets.copy(), **kwargs
        )
        current = decode_detections(
            dets=dets.copy(),
            outside_center_decode_mode="legacy",
            **kwargs
        )
        np.testing.assert_allclose(
            np.asarray(original[11]),
            np.asarray(current[11]),
            rtol=0,
            atol=0,
        )

    def test_four_modes_share_topk_and_non_center_fields(self):
        outputs = make_outputs(
            center=(0.0, 0.5), offset=(-0.2, 0.0)
        )
        extracted = {}
        debug = {}
        for mode in (
                "full", "zero_offset_new", "legacy", "full_legacy_ry"):
            extracted[mode], debug[mode] = extract_dets_from_outputs(
                outputs,
                topk=1,
                outside_center_decode_mode=mode,
                return_debug=True,
            )

        for mode in ("zero_offset_new", "legacy", "full_legacy_ry"):
            torch.testing.assert_close(
                extracted["full"][:, :, :34],
                extracted[mode][:, :, :34],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                debug["full"]["query_ids"],
                debug[mode]["query_ids"],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                debug["full"]["angle_head"],
                debug[mode]["angle_head"],
                rtol=0,
                atol=0,
            )

        torch.testing.assert_close(
            debug["full"]["decoded_center_norm"],
            debug["full_legacy_ry"]["decoded_center_norm"],
            rtol=0,
            atol=0,
            check_stride=False,
        )
        torch.testing.assert_close(
            debug["full"]["decoded_center_norm"]
            - debug["zero_offset_new"]["decoded_center_norm"],
            torch.tensor([[[-0.2, 0.0]]]),
            rtol=0,
            atol=1e-7,
            check_stride=False,
        )
        torch.testing.assert_close(
            debug["zero_offset_new"]["decoded_center_norm"],
            debug["zero_offset_new"]["representative_center_norm"],
            rtol=0,
            atol=0,
            check_stride=False,
        )

    def test_zero_offset_new_uses_new_ry_path_not_legacy(self):
        calibration = make_calibration()
        outputs = make_outputs(
            center=(0.4, 0.6), offset=(0.2, 0.0)
        )
        zero_dets = extract_dets_from_outputs(
            outputs,
            topk=1,
            outside_center_decode_mode="zero_offset_new",
        ).numpy()
        legacy_dets = extract_dets_from_outputs(
            outputs,
            topk=1,
            outside_center_decode_mode="legacy",
        ).numpy()
        info = {
            "img_size": np.array([[1280.0, 384.0]], dtype=np.float32),
            "img_id": np.array([19]),
        }
        kwargs = dict(
            info=info,
            calibs=[calibration],
            cls_mean_size=np.zeros((3, 3), dtype=np.float32),
            threshold=0.0,
        )
        zero = decode_detections(
            zero_dets.copy(),
            outside_center_decode_mode="zero_offset_new",
            **kwargs
        )[19][0]
        legacy = decode_detections(
            legacy_dets.copy(),
            outside_center_decode_mode="legacy",
            **kwargs
        )[19][0]
        np.testing.assert_allclose(zero[2:12], legacy[2:12], rtol=0, atol=0)
        self.assertNotEqual(zero[12], legacy[12])
        self.assertAlmostEqual(
            zero[12],
            calibration.alpha2ry(0.0, 0.4 * 1280.0),
            places=6,
        )

    def test_full_legacy_ry_uses_full_location_and_legacy_orientation(self):
        calibration = make_calibration()
        outputs = make_outputs(
            center=(0.4, 0.6), offset=(0.2, -0.1)
        )
        info = {
            "img_size": np.array([[1280.0, 384.0]], dtype=np.float32),
            "img_id": np.array([23]),
        }
        kwargs = dict(
            info=info,
            calibs=[calibration],
            cls_mean_size=np.zeros((3, 3), dtype=np.float32),
            threshold=0.0,
        )
        decoded = {}
        for mode in ("full", "legacy", "full_legacy_ry"):
            dets = extract_dets_from_outputs(
                outputs,
                topk=1,
                outside_center_decode_mode=mode,
            ).numpy()
            decoded[mode] = decode_detections(
                dets.copy(),
                outside_center_decode_mode=mode,
                **kwargs
            )[23][0]

        np.testing.assert_allclose(
            decoded["full_legacy_ry"][2:12],
            decoded["full"][2:12],
            rtol=0,
            atol=0,
        )
        self.assertAlmostEqual(
            decoded["full_legacy_ry"][12],
            decoded["legacy"][12],
            places=6,
        )
        self.assertNotEqual(
            decoded["full_legacy_ry"][12],
            decoded["full"][12],
        )

    def test_legacy_does_not_read_offset_output(self):
        class OffsetAccessFails(dict):
            def __getitem__(self, key):
                if key == "pred_outside_center_offset":
                    raise AssertionError("legacy accessed offset output")
                return super().__getitem__(key)

        outputs = OffsetAccessFails(
            make_outputs(offset=(0.3, -0.1))
        )
        detections = extract_dets_from_outputs(
            outputs,
            topk=1,
            outside_center_decode_mode="legacy",
        )
        self.assertTrue(torch.isfinite(detections).all())

    def test_decode_mode_validation_and_disabled_resolution(self):
        self.assertEqual(
            resolve_outside_center_decode_mode(True, None), "full"
        )
        self.assertEqual(
            resolve_outside_center_decode_mode(False, "full"), "legacy"
        )
        with self.assertRaisesRegex(ValueError, "Invalid"):
            validate_outside_center_decode_mode("not_a_mode")
        with self.assertRaisesRegex(ValueError, "Invalid"):
            resolve_outside_center_decode_mode(True, "not_a_mode")

    def test_ablation_configs_only_differ_by_mode_and_tag(self):
        paths = (
            "configs/eval_outside_A_full.yaml",
            "configs/eval_outside_B_zero.yaml",
            "configs/eval_outside_C_legacy.yaml",
            "configs/eval_outside_D_full_legacy_ry.yaml",
        )
        configs = []
        for path in paths:
            with open(path, "r") as config_file:
                configs.append(
                    yaml.load(config_file, Loader=yaml.Loader)
                )

        expected = (
            ("full", "A_full"),
            ("zero_offset_new", "B_zero_offset_new"),
            ("legacy", "C_legacy"),
            ("full_legacy_ry", "D_full_legacy_ry"),
        )
        checkpoint = (
            "outputs/monodetr_outside_center_0.00015_test/"
            "checkpoint_best.pth"
        )
        normalized = []
        for config, (mode, tag) in zip(configs, expected):
            self.assertTrue(
                config["dataset"]["use_outside_center_modeling"]
            )
            self.assertEqual(
                config["tester"]["outside_center_decode_mode"], mode
            )
            self.assertEqual(config["tester"]["result_tag"], tag)
            self.assertEqual(
                config["tester"]["checkpoint_path"], checkpoint
            )
            copy = dict(config)
            copy["tester"] = dict(copy["tester"])
            copy["tester"].pop("outside_center_decode_mode")
            copy["tester"].pop("result_tag")
            normalized.append(copy)
        self.assertEqual(normalized[0], normalized[1])
        self.assertEqual(normalized[1], normalized[2])
        self.assertEqual(normalized[2], normalized[3])

    def test_select_projected_center_manual_ablation(self):
        representative = torch.tensor([[0.0, 0.5]])
        offset = torch.tensor([[-0.2, 0.0]])
        full, full_used = select_projected_center(
            representative, offset, "full"
        )
        zero, zero_used = select_projected_center(
            representative, offset, "zero_offset_new"
        )
        legacy, legacy_used = select_projected_center(
            representative, None, "legacy"
        )
        full_legacy_ry, full_legacy_ry_used = select_projected_center(
            representative, offset, "full_legacy_ry"
        )
        torch.testing.assert_close(
            full, torch.tensor([[-0.2, 0.5]]), rtol=0, atol=0
        )
        torch.testing.assert_close(
            zero, representative, rtol=0, atol=0
        )
        torch.testing.assert_close(
            legacy, representative, rtol=0, atol=0
        )
        torch.testing.assert_close(
            full_legacy_ry,
            torch.tensor([[-0.2, 0.5]]),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(full_used, offset, rtol=0, atol=0)
        torch.testing.assert_close(
            full_legacy_ry_used, offset, rtol=0, atol=0
        )
        torch.testing.assert_close(
            zero_used, torch.zeros_like(offset), rtol=0, atol=0
        )
        torch.testing.assert_close(
            legacy_used, torch.zeros_like(representative), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
