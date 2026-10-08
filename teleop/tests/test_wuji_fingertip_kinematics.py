import unittest
from pathlib import Path
import sys

import numpy as np


TELEOP_ROOT = Path(__file__).resolve().parents[1]
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

from utils.wuji_fingertip_kinematics import (
    FINGER_NAMES,
    WujiFingertipKinematics,
    estimate_wuji_fingertip_positions,
    estimate_wuji_keypoint_positions,
)
from utils.wuji_keypoint_visualization import (
    DEFAULT_KEYPOINT_SPECS,
    generate_wuji_keypoint_html,
)


class WujiFingertipKinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kinematics = WujiFingertipKinematics()
        cls.arm_q_deg = np.array([0.0, 30.0, 0.0, 90.0, 0.0, -30.0, 0.0])
        cls.hand_q = np.zeros((5, 4), dtype=np.float64)

    def test_returns_five_finite_base_frame_positions(self):
        positions = self.kinematics.estimate(
            self.arm_q_deg, self.hand_q, flange_thickness_m=0.026
        )

        self.assertEqual(tuple(positions), FINGER_NAMES)
        for position in positions.values():
            self.assertEqual(position.shape, (3,))
            self.assertTrue(np.all(np.isfinite(position)))

    def test_arm_degree_and_radian_inputs_match(self):
        degree_result = self.kinematics.estimate(
            self.arm_q_deg, self.hand_q, 0.026, arm_angle_unit="degree"
        )
        radian_result = self.kinematics.estimate(
            np.deg2rad(self.arm_q_deg), self.hand_q, 0.026, arm_angle_unit="radian"
        )

        for finger in FINGER_NAMES:
            np.testing.assert_allclose(degree_result[finger], radian_result[finger], atol=1e-12)

    def test_public_arm_link_transform_matches_model_transform(self):
        direct = self.kinematics.arm_link_transform(self.arm_q_deg, "link_7")
        model = self.kinematics.model_transforms(
            self.arm_q_deg, self.hand_q, 0.0
        )["arm"]["link_7"]

        np.testing.assert_allclose(direct, model, atol=1e-12)

    def test_one_finger_joint_only_moves_that_fingertip(self):
        initial = self.kinematics.estimate(self.arm_q_deg, self.hand_q, 0.026)
        moved_q = self.hand_q.copy()
        moved_q[1, 2] = 0.5
        moved = self.kinematics.estimate(self.arm_q_deg, moved_q, 0.026)

        self.assertGreater(np.linalg.norm(moved["index"] - initial["index"]), 1e-4)
        for finger in ("thumb", "middle", "ring", "little"):
            np.testing.assert_allclose(moved[finger], initial[finger], atol=1e-12)

    def test_flange_thickness_rigidly_translates_all_fingertips(self):
        first = self.kinematics.estimate(self.arm_q_deg, self.hand_q, 0.010)
        second = self.kinematics.estimate(self.arm_q_deg, self.hand_q, 0.035)
        shifts = np.stack([second[name] - first[name] for name in FINGER_NAMES])

        np.testing.assert_allclose(shifts, np.repeat(shifts[:1], 5, axis=0), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(shifts, axis=1), 0.025, atol=1e-12)

    def test_local_fingertip_offsets_are_supported(self):
        initial = self.kinematics.estimate(self.arm_q_deg, self.hand_q, 0.026)
        offsets = np.zeros((5, 3), dtype=np.float64)
        offsets[0] = [0.0, 0.0, 0.01]
        shifted = self.kinematics.estimate(
            self.arm_q_deg, self.hand_q, 0.026, fingertip_offsets_m=offsets
        )

        self.assertAlmostEqual(
            np.linalg.norm(shifted["thumb"] - initial["thumb"]), 0.01
        )
        for finger in ("index", "middle", "ring", "little"):
            np.testing.assert_allclose(shifted[finger], initial[finger], atol=1e-12)

    def test_mount_to_palm_calibration_is_supported(self):
        initial = self.kinematics.estimate(self.arm_q_deg, self.hand_q, 0.026)
        mount_to_palm = np.eye(4, dtype=np.float64)
        mount_to_palm[0, 3] = 0.012
        shifted = self.kinematics.estimate(
            self.arm_q_deg,
            self.hand_q,
            0.026,
            mount_to_palm_transform=mount_to_palm,
        )
        shifts = np.stack(
            [shifted[name] - initial[name] for name in FINGER_NAMES]
        )

        np.testing.assert_allclose(shifts, np.repeat(shifts[:1], 5, axis=0), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(shifts, axis=1), 0.012, atol=1e-12)

    def test_convenience_function_uses_repository_models(self):
        positions = estimate_wuji_fingertip_positions(
            self.arm_q_deg, self.hand_q.reshape(20), 0.026
        )
        self.assertEqual(tuple(positions), FINGER_NAMES)

    def test_estimates_all_configured_keypoints(self):
        positions = self.kinematics.estimate_keypoints(
            self.arm_q_deg,
            self.hand_q,
            0.026,
            keypoint_specs=DEFAULT_KEYPOINT_SPECS,
        )

        expected_names = tuple(spec["name"] for spec in DEFAULT_KEYPOINT_SPECS)
        self.assertEqual(tuple(positions), expected_names)
        self.assertEqual(len(positions), len(DEFAULT_KEYPOINT_SPECS))
        self.assertTrue(
            {
                "thumb_2",
                "thumb_3",
                "index_2",
                "index_3",
                "middle_2",
                "middle_3",
                "ring_2",
                "ring_3",
                "little_2",
                "little_3",
            }.issubset(positions)
        )
        for position in positions.values():
            self.assertEqual(position.shape, (3,))
            self.assertTrue(np.all(np.isfinite(position)))

        transforms = self.kinematics.model_transforms(
            self.arm_q_deg, self.hand_q, 0.026
        )
        thumb_3_spec = next(
            spec for spec in DEFAULT_KEYPOINT_SPECS if spec["name"] == "thumb_3"
        )
        base_to_link = transforms["hand"][thumb_3_spec["link_name"]]
        expected = (
            base_to_link[:3, 3]
            + base_to_link[:3, :3] @ np.asarray(thumb_3_spec["offset_m"])
        )
        np.testing.assert_allclose(positions["thumb_3"], expected, atol=1e-12)

    def test_keypoint_convenience_function_uses_configured_names(self):
        positions = estimate_wuji_keypoint_positions(
            self.arm_q_deg,
            self.hand_q,
            0.026,
            keypoint_specs=DEFAULT_KEYPOINT_SPECS,
        )

        self.assertEqual(
            tuple(positions),
            tuple(spec["name"] for spec in DEFAULT_KEYPOINT_SPECS),
        )

    def test_invalid_keypoint_specs_are_rejected(self):
        invalid_specs = (
            {"name": "unknown", "link_name": "missing_link"},
            {"name": "bad_offset", "link_name": "palm_link", "offset_m": (0, 0)},
        )
        for spec in invalid_specs:
            with self.subTest(spec=spec):
                with self.assertRaises(ValueError):
                    self.kinematics.estimate_keypoints(
                        self.arm_q_deg,
                        self.hand_q,
                        0.026,
                        keypoint_specs=(spec,),
                    )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.kinematics.estimate(np.zeros(6), self.hand_q, 0.026)
        with self.assertRaises(ValueError):
            self.kinematics.estimate(self.arm_q_deg, np.zeros(19), 0.026)
        with self.assertRaises(ValueError):
            self.kinematics.estimate(self.arm_q_deg, np.zeros((4, 5)), 0.026)
        with self.assertRaises(ValueError):
            self.kinematics.estimate(self.arm_q_deg, self.hand_q, -0.001)
        with self.assertRaises(ValueError):
            self.kinematics.estimate(
                self.arm_q_deg, self.hand_q, 0.026, arm_angle_unit="degrees"
            )

    def test_html_snapshot_contains_model_and_keypoints(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            output = generate_wuji_keypoint_html(
                Path(temp_dir) / "keypoints.html",
                self.arm_q_deg,
                self.hand_q,
                0.026,
                kinematics=self.kinematics,
            )
            html = output.read_text(encoding="utf-8")

        self.assertIn('"frame":"RM75 base_link"', html)
        self.assertIn('"name":"thumb_tip"', html)
        self.assertIn('"name":"little_tip"', html)
        self.assertIn('"finger1_tip_link"', html)
        self.assertIn("three@0.160.0", html)
        self.assertIn('id="rotate-mode"', html)
        self.assertIn('id="pan-mode"', html)
        self.assertIn("controls.mouseButtons.LEFT = panMode ? THREE.MOUSE.PAN", html)
        self.assertIn("panX += deltaX", html)

    def test_html_snapshot_accepts_additional_keypoint(self):
        import tempfile

        extra_point = {
            "name": "index_distal_joint",
            "link_name": "finger2_link4",
            "offset_m": (0.0, 0.0, 0.005),
            "color": "#2f855a",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = generate_wuji_keypoint_html(
                Path(temp_dir) / "keypoints.html",
                self.arm_q_deg,
                self.hand_q,
                0.026,
                keypoint_specs=DEFAULT_KEYPOINT_SPECS + (extra_point,),
                kinematics=self.kinematics,
            )
            html = output.read_text(encoding="utf-8")

        self.assertIn('"name":"index_distal_joint"', html)
        self.assertIn('"link_name":"finger2_link4"', html)
        self.assertIn('"offset_m":[0.0,0.0,0.005]', html)


def _run_live_measurement():
    import argparse

    parser = argparse.ArgumentParser(
        description="Print live Wuji keypoint positions in the RM75 base frame"
    )
    parser.add_argument("--arm-ip", default="192.168.1.18")
    parser.add_argument("--arm-port", type=int, default=8080)
    parser.add_argument(
        "--hand-device",
        default=None,
        help="Wuji USB serial number; omit to select the connected right hand",
    )
    parser.add_argument(
        "--flange-thickness",
        type=float,
        required=True,
        help="link_7 to Wuji mounting-plane distance along local +Z, in meters",
    )
    parser.add_argument(
        "--html-output",
        default="wuji_fingertip_keypoints.html",
        help="output path for the interactive keypoint HTML snapshot",
    )
    args = parser.parse_args()

    arm = None
    hand = None
    try:
        import wujihandpy

        from utils.Realman_controller import RoboticArmController

        arm = RoboticArmController(ip=args.arm_ip, port=args.arm_port)
        if args.hand_device:
            hand = wujihandpy.Hand(serial_number=args.hand_device)
        else:
            hand = wujihandpy.Hand(side="right")

        if int(hand.read_handedness()) != 0:
            raise RuntimeError("Connected Wuji device is not a right hand")

        arm_ok, arm_q_deg = arm.get_joint_state()
        if not arm_ok:
            raise RuntimeError("Failed to read current RM75 joint state")
        hand_q_rad = np.asarray(
            hand.read_joint_actual_position(), dtype=np.float64
        )

        kinematics = WujiFingertipKinematics()
        positions = kinematics.estimate_keypoints(
            arm_q_deg,
            hand_q_rad,
            flange_thickness_m=args.flange_thickness,
            keypoint_specs=DEFAULT_KEYPOINT_SPECS,
        )

        np.set_printoptions(precision=6, suppress=True)
        print("\nCurrent measured joint state")
        print(f"  RM75 [deg]: {np.asarray(arm_q_deg, dtype=np.float64)}")
        print("  Wuji [rad], rows thumb/index/middle/ring/little:")
        print(hand_q_rad)
        print(
            f"\nKeypoint positions relative to RM75 base_link "
            f"(flange thickness={args.flange_thickness:.6f} m)"
        )
        print("  keypoint        x [m]       y [m]       z [m]"
              "       x [mm]      y [mm]      z [mm]")
        for keypoint_name, xyz_m in positions.items():
            xyz_mm = xyz_m * 1000.0
            print(
                f"  {keypoint_name:<11}  "
                f"{xyz_m[0]:+10.6f}  {xyz_m[1]:+10.6f}  {xyz_m[2]:+10.6f}"
                f"   {xyz_mm[0]:+10.2f}  {xyz_mm[1]:+10.2f}  {xyz_mm[2]:+10.2f}"
            )
        html_path = generate_wuji_keypoint_html(
            args.html_output,
            arm_q_deg,
            hand_q_rad,
            args.flange_thickness,
            kinematics=kinematics,
        )
        print(f"\n[HTML] Interactive keypoint view: {html_path}")
        return 0
    except Exception as exc:
        print(f"[FAIL] Live keypoint measurement failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        hand = None
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(_run_live_measurement())
