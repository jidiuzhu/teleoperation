import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import yaml

from utils.wuji_retargeting import (
    DEFAULT_INTERNAL_LINK_NAMES,
    FINGER_INTERNAL_INDICES,
    FINGER_MCP_INDICES,
    FINGER_TIP_INDICES,
    WujiDexPilotRetargeter,
    load_fingertip_scales,
    load_internal_constraint_config,
    scale_finger_targets_from_mcp,
    scale_fingertips_from_mcp,
)


TELEOP_ROOT = Path(__file__).resolve().parents[1]


class WujiFingertipScaleTest(unittest.TestCase):
    def test_defaults_preserve_landmarks(self):
        landmarks = np.arange(75, dtype=np.float64).reshape(25, 3) / 100.0
        scales = load_fingertip_scales({})

        corrected = scale_fingertips_from_mcp(landmarks, scales)

        np.testing.assert_allclose(scales, np.ones(5))
        np.testing.assert_allclose(corrected, landmarks)
        self.assertIsNot(corrected, landmarks)

    def test_each_scale_only_changes_its_fingertip(self):
        landmarks = np.zeros((25, 3), dtype=np.float64)
        for finger, (mcp_index, tip_index) in enumerate(
            zip(FINGER_MCP_INDICES, FINGER_TIP_INDICES)
        ):
            landmarks[mcp_index] = [finger, 1.0, 0.0]
            landmarks[tip_index] = [finger + 1.0, 1.0, 0.0]

        scales = load_fingertip_scales(
            {
                "wuji_morphology": {
                    "fingertip_scale": {
                        "thumb": 1.0,
                        "index": 1.1,
                        "middle": 1.2,
                        "ring": 1.3,
                        "little": 1.4,
                    }
                }
            }
        )
        corrected = scale_fingertips_from_mcp(landmarks, scales)

        expected_vectors = scales[:, None] * np.array([[1.0, 0.0, 0.0]])
        actual_vectors = (
            corrected[FINGER_TIP_INDICES] - corrected[FINGER_MCP_INDICES]
        )
        np.testing.assert_allclose(actual_vectors, expected_vectors)

        untouched = np.ones(25, dtype=bool)
        untouched[FINGER_TIP_INDICES] = False
        np.testing.assert_allclose(corrected[untouched], landmarks[untouched])

    def test_partial_config_defaults_other_fingers(self):
        scales = load_fingertip_scales(
            {"wuji_morphology": {"fingertip_scale": {"little": 1.25}}}
        )
        np.testing.assert_allclose(scales, [1.0, 1.0, 1.0, 1.0, 1.25])

    def test_pairwise_vectors_use_consistent_corrected_tips(self):
        landmarks = np.zeros((25, 3), dtype=np.float64)
        for finger, (mcp_index, tip_index) in enumerate(
            zip(FINGER_MCP_INDICES, FINGER_TIP_INDICES)
        ):
            landmarks[mcp_index] = [0.01 * finger, 0.0, 0.0]
            landmarks[tip_index] = [0.01 * finger, 0.08, 0.0]

        scales = np.asarray([1.0, 1.1, 1.0, 1.2, 1.4])
        corrected = scale_fingertips_from_mcp(landmarks, scales)
        pairwise_origin = np.asarray([9, 14, 19, 24])
        pairwise_task = np.asarray([4, 4, 4, 4])

        reference = corrected[pairwise_task] - corrected[pairwise_origin]

        np.testing.assert_allclose(
            reference,
            corrected[FINGER_TIP_INDICES[0]]
            - corrected[FINGER_TIP_INDICES[1:]],
        )

    def test_invalid_scale_and_unknown_name_are_rejected(self):
        invalid_configs = [
            {"wuji_morphology": {"fingertip_scale": {"little": 0.1}}},
            {"wuji_morphology": {"fingertip_scale": {"little": "bad"}}},
            {"wuji_morphology": {"fingertip_scale": {"pinky": 1.1}}},
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                load_fingertip_scales(config)

    def test_internal_landmarks_and_tips_share_each_finger_scale(self):
        landmarks = np.zeros((25, 3), dtype=np.float64)
        for finger, (mcp_index, internal_index, tip_index) in enumerate(
            zip(
                FINGER_MCP_INDICES,
                FINGER_INTERNAL_INDICES,
                FINGER_TIP_INDICES,
            )
        ):
            landmarks[mcp_index] = [0.01 * finger, 0.0, 0.0]
            landmarks[internal_index] = [0.01 * finger, 0.04, 0.01]
            landmarks[tip_index] = [0.01 * finger, 0.08, 0.02]

        scales = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4])
        corrected = scale_finger_targets_from_mcp(landmarks, scales)

        expected_internal = landmarks[FINGER_MCP_INDICES] + scales[:, None] * (
            landmarks[FINGER_INTERNAL_INDICES] - landmarks[FINGER_MCP_INDICES]
        )
        expected_tips = landmarks[FINGER_MCP_INDICES] + scales[:, None] * (
            landmarks[FINGER_TIP_INDICES] - landmarks[FINGER_MCP_INDICES]
        )
        np.testing.assert_allclose(
            corrected[FINGER_INTERNAL_INDICES], expected_internal
        )
        np.testing.assert_allclose(corrected[FINGER_TIP_INDICES], expected_tips)


class WujiInternalConstraintConfigTest(unittest.TestCase):
    def test_defaults_enable_all_five_fingers(self):
        constraint = load_internal_constraint_config({})

        self.assertTrue(constraint["enabled"])
        self.assertEqual(
            constraint["internal_link_names"], DEFAULT_INTERNAL_LINK_NAMES
        )
        np.testing.assert_allclose(
            constraint["wrist_to_joint_weights"], np.full(5, 1.0)
        )
        np.testing.assert_allclose(
            constraint["joint_to_tip_weights"], np.full(5, 2.0)
        )

    def test_per_finger_weights_and_disable_switch_are_loaded(self):
        constraint = load_internal_constraint_config(
            {
                "wuji_internal_constraints": {
                    "enabled": False,
                    "wrist_to_joint_weight": {"thumb": 1.5},
                    "joint_to_tip_weight": {"little": 7.0},
                }
            }
        )

        self.assertFalse(constraint["enabled"])
        np.testing.assert_allclose(
            constraint["wrist_to_joint_weights"], [1.5, 1.0, 1.0, 1.0, 1.0]
        )
        np.testing.assert_allclose(
            constraint["joint_to_tip_weights"], [2.0, 2.0, 2.0, 2.0, 7.0]
        )

    def test_invalid_internal_constraint_fields_are_rejected(self):
        invalid_configs = [
            {"wuji_internal_constraints": {"enabled": "yes"}},
            {"wuji_internal_constraints": {"internal_link_names": ["one"]}},
            {
                "wuji_internal_constraints": {
                    "wrist_to_joint_weight": {"little": -1.0}
                }
            },
            {
                "wuji_internal_constraints": {
                    "joint_to_tip_weight": {"pinky": 5.0}
                }
            },
        ]
        for config in invalid_configs:
            with self.subTest(config=config), self.assertRaises(ValueError):
                load_internal_constraint_config(config)


class WujiDexPilotIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = TELEOP_ROOT / "teleop" / "wuji.yml"
        cls.retargeter = WujiDexPilotRetargeter(config_path)
        cls.temp_dir = TemporaryDirectory()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        config["wuji_internal_constraints"]["enabled"] = True
        enabled_config = Path(cls.temp_dir.name) / "wuji_enabled.yml"
        enabled_config.write_text(yaml.safe_dump(config), encoding="utf-8")
        cls.enabled_retargeter = WujiDexPilotRetargeter(enabled_config)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_default_config_matches_original_fifteen_vector_dexpilot(self):
        self.assertFalse(self.retargeter.internal_constraints["enabled"])
        self.assertEqual(self.retargeter._retargeting.optimizer.target_link_human_indices.shape, (2, 15))

    def test_optimizer_has_original_and_internal_vectors(self):
        optimizer = self.enabled_retargeter._retargeting.optimizer

        self.assertEqual(optimizer.base_vector_count, 15)
        self.assertEqual(optimizer.internal_vector_count, 10)
        self.assertEqual(optimizer.target_link_human_indices.shape, (2, 25))
        np.testing.assert_allclose(
            optimizer.internal_constraint_weights,
            [1.5, 0.5, 0.5, 0.5, 0.3, 1.7, 0.5, 0.5, 0.5, 0.3],
        )
        np.testing.assert_array_equal(
            optimizer.target_link_human_indices[0, 15:],
            np.concatenate([np.zeros(5, dtype=int), FINGER_INTERNAL_INDICES]),
        )
        np.testing.assert_array_equal(
            optimizer.target_link_human_indices[1, 15:],
            np.concatenate([FINGER_INTERNAL_INDICES, FINGER_TIP_INDICES]),
        )
        self.assertEqual(
            optimizer.origin_link_names[15:],
            ["palm_link"] * 5 + list(self.enabled_retargeter.internal_constraints["internal_link_names"]),
        )
        self.assertEqual(
            optimizer.task_link_names[15:],
            list(self.enabled_retargeter.internal_constraints["internal_link_names"])
            + [f"finger{finger}_tip_link" for finger in range(1, 6)],
        )

    def test_internal_vectors_are_geometrically_consistent(self):
        landmarks = np.arange(75, dtype=np.float64).reshape(25, 3) / 1000.0
        corrected = scale_finger_targets_from_mcp(
            landmarks, self.retargeter.fingertip_scales
        )

        reference = self.enabled_retargeter.build_reference(landmarks)

        self.assertEqual(reference.shape, (25, 3))
        np.testing.assert_allclose(
            reference[15:20],
            corrected[FINGER_INTERNAL_INDICES] - corrected[0],
        )
        np.testing.assert_allclose(
            reference[20:25],
            corrected[FINGER_TIP_INDICES] - corrected[FINGER_INTERNAL_INDICES],
        )
        np.testing.assert_allclose(
            reference[15:20] + reference[20:25],
            corrected[FINGER_TIP_INDICES] - corrected[0],
        )

    def test_real_config_produces_finite_hardware_ordered_output(self):
        landmarks = np.zeros((25, 3), dtype=np.float64)
        for finger, (mcp_index, internal_index, tip_index) in enumerate(
            zip(
                FINGER_MCP_INDICES,
                FINGER_INTERNAL_INDICES,
                FINGER_TIP_INDICES,
            )
        ):
            x = 0.025 * (finger - 2)
            landmarks[mcp_index] = [x, 0.015, 0.0]
            landmarks[internal_index] = [x, 0.055, 0.0]
            landmarks[tip_index] = [x, 0.095, 0.0]

        target = self.retargeter.retarget(landmarks)

        self.assertEqual(target.shape, (20,))
        self.assertTrue(np.all(np.isfinite(target)))


if __name__ == "__main__":
    unittest.main()
