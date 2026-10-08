from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


TELEOP_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = TELEOP_ROOT / "scripts"
for import_path in (TELEOP_ROOT, SCRIPTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import deploy_actbh_wuji as deployment
from utils.actbh_observation import (
    ACTBHObservationConfig,
    DEPTH_IMAGE_KEY,
    ENVIRONMENT_STATE_KEY,
    POINT_CLOUD_KEY,
    RGB_KEY,
    STATE_KEY,
    LiveACTBHObservationBuilder,
    assemble_actbh_observation,
)


class FakeKeypointKinematics:
    def estimate_keypoints(self, *_args, keypoint_specs, **_kwargs):
        return {
            spec["name"]: np.zeros(3, dtype=np.float64)
            for spec in keypoint_specs
        }


class ACTBHObservationTest(unittest.TestCase):
    def test_assembled_observation_matches_model_contract(self):
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]
        config = ACTBHObservationConfig(
            image_height=8,
            image_width=12,
            point_count=5,
        )

        observation = assemble_actbh_observation(
            rgb_hwc=np.zeros((4, 6, 3), dtype=np.uint8),
            depth_mm=np.full((4, 6), 825, dtype=np.uint16),
            arm_q_rad=np.arange(7, dtype=np.float32),
            hand_q_rad=np.arange(20, dtype=np.float32),
            distances_m=np.linspace(0.01, 0.15, 15),
            pointcloud_camera=np.array(
                [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [99, 99, 99]],
                dtype=np.float32,
            ),
            pointcloud_count=2,
            T_base_camera=transform,
            config=config,
        )

        self.assertEqual(observation[RGB_KEY].shape, (8, 12, 3))
        self.assertEqual(observation[DEPTH_IMAGE_KEY].shape, (8, 12, 3))
        self.assertEqual(observation[STATE_KEY].shape, (27,))
        self.assertEqual(observation[ENVIRONMENT_STATE_KEY].shape, (15,))
        self.assertEqual(observation[POINT_CLOUD_KEY].shape, (5, 3))
        np.testing.assert_allclose(observation[ENVIRONMENT_STATE_KEY], np.linspace(0.01, 0.15, 15))
        np.testing.assert_allclose(observation[POINT_CLOUD_KEY][0], [1.0, 2.0, 3.5])
        self.assertFalse(np.any(observation[POINT_CLOUD_KEY] == 99))

    def test_live_builder_converts_bgr_and_computes_15_distances(self):
        specs = tuple({"name": f"point_{index}"} for index in range(15))
        builder = LiveACTBHObservationBuilder(
            T_base_camera=np.eye(4),
            keypoint_kinematics=FakeKeypointKinematics(),
            keypoint_specs=specs,
            flange_thickness_m=0.036,
            config=ACTBHObservationConfig(
                image_height=2,
                image_width=2,
                point_count=4,
            ),
        )
        bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        bgr[..., 0] = 255
        frame = {
            "valid": True,
            "color": bgr,
            "depth": np.full((2, 2), 500, dtype=np.uint16),
            "pointcloud_xyz": np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            "pointcloud_count": 1,
        }

        observation = builder.build(np.zeros(7), np.zeros(20), frame)

        self.assertEqual(int(observation[RGB_KEY][0, 0, 0]), 0)
        self.assertEqual(int(observation[RGB_KEY][0, 0, 2]), 255)
        np.testing.assert_allclose(observation[ENVIRONMENT_STATE_KEY], np.ones(15))

    def test_live_builder_rejects_stale_frame(self):
        specs = tuple({"name": f"point_{index}"} for index in range(15))
        builder = LiveACTBHObservationBuilder(
            T_base_camera=np.eye(4),
            keypoint_kinematics=FakeKeypointKinematics(),
            keypoint_specs=specs,
            flange_thickness_m=0.036,
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            builder.build(np.zeros(7), np.zeros(20), {"valid": False})


class MotionSafetyFilterTest(unittest.TestCase):
    def setUp(self):
        self.filter = deployment.MotionSafetyFilter(
            deployment.MotionSafetyConfig(
                arm_limits_rad=np.tile([-1.0, 1.0], (7, 1)),
                hand_limits_rad=np.tile([-0.5, 0.5], (20, 1)),
                arm_max_step_rad=0.1,
                hand_max_step_rad=0.05,
                arm_tracking_error_rad=0.2,
                arm_alpha=1.0,
                hand_alpha=1.0,
            )
        )

    def test_limits_absolute_action_and_per_step_change(self):
        current_arm = np.zeros(7)
        current_hand = np.zeros(20)
        self.filter.reset(current_arm, current_hand)

        safe, diagnostics = self.filter.filter(
            np.full(27, 10.0),
            current_arm,
            current_hand,
            enforce_tracking=True,
        )

        np.testing.assert_allclose(safe[:7], 0.1)
        np.testing.assert_allclose(safe[7:], 0.05)
        self.assertTrue(diagnostics["arm_limit_clipped"])
        self.assertTrue(diagnostics["hand_limit_clipped"])

    def test_tracking_error_stops_motion(self):
        self.filter.reset(np.zeros(7), np.zeros(20))
        with self.assertRaisesRegex(deployment.SafetyStop, "tracking error"):
            self.filter.filter(
                np.zeros(27),
                np.full(7, 0.3),
                np.zeros(20),
                enforce_tracking=True,
            )

    def test_non_finite_action_is_rejected(self):
        action = np.zeros(27)
        action[3] = np.nan
        with self.assertRaisesRegex(deployment.SafetyStop, "finite shape"):
            self.filter.filter(
                action,
                np.zeros(7),
                np.zeros(20),
                enforce_tracking=False,
            )


class DeploymentArgumentsTest(unittest.TestCase):
    def test_motion_requires_explicit_confirmation(self):
        with self.assertRaises(SystemExit):
            deployment.parse_args(["--enable-motion"])

        args = deployment.parse_args(
            [
                "--enable-motion",
                f"--motion-confirmation={deployment.MOTION_CONFIRMATION}",
            ]
        )
        self.assertTrue(args.enable_motion)

    def test_replay_cannot_enable_motion(self):
        with self.assertRaises(SystemExit):
            deployment.parse_args(
                [
                    "--replay-h5=episode.h5",
                    "--enable-motion",
                    f"--motion-confirmation={deployment.MOTION_CONFIRMATION}",
                ]
            )

    def test_live_dependency_check_fails_before_hardware_imports(self):
        with patch.object(deployment.importlib.util, "find_spec", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "robotic-arm==1.1.6"):
                deployment.check_live_dependencies("wujihandpy")


if __name__ == "__main__":
    unittest.main()
