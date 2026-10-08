import unittest
from unittest.mock import Mock
from queue import Queue

import numpy as np

from scripts.teleop_real_server_wuji import TeleopServer


class WujiCameraModeTest(unittest.TestCase):
    def make_server(self, mode):
        server = TeleopServer.__new__(TeleopServer)
        server.camera_mode = mode
        server.enabled_camera_names = {
            "both": ("D405", "ZED_M"),
            "d405": ("D405",),
            "zed": ("ZED_M",),
            "none": (),
        }[mode]
        server.preview_camera = (
            "ZED_M"
            if "ZED_M" in server.enabled_camera_names
            else "D405"
            if "D405" in server.enabled_camera_names
            else "none"
        )
        return server

    def test_preview_selection_prefers_zed_when_both_are_enabled(self):
        server = self.make_server("both")
        d405_bgr = np.zeros((2, 3, 3), dtype=np.uint8)
        zed_left_rgb = np.full((2, 4, 3), 10, dtype=np.uint8)
        zed_right_rgb = np.full((2, 4, 3), 20, dtype=np.uint8)
        frames = {
            "D405": {"color": d405_bgr},
            "ZED_M": {
                "color": zed_left_rgb,
                "color_right": zed_right_rgb,
            },
        }

        preview = server._build_preview_image(frames)

        self.assertEqual(preview.shape, (2, 8, 3))
        np.testing.assert_array_equal(preview[:, :4], zed_left_rgb)
        np.testing.assert_array_equal(preview[:, 4:], zed_right_rgb)

    def test_d405_preview_is_converted_to_rgb_and_duplicated(self):
        server = self.make_server("d405")
        d405_bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)

        preview = server._build_preview_image(
            {"D405": {"color": d405_bgr}}
        )

        expected_eye = np.array([[[3, 2, 1]]], dtype=np.uint8)
        self.assertEqual(preview.shape, (1, 2, 3))
        np.testing.assert_array_equal(preview[:, :1], expected_eye)
        np.testing.assert_array_equal(preview[:, 1:], expected_eye)

    def test_control_only_mode_has_no_preview(self):
        server = self.make_server("none")
        self.assertIsNone(server._build_preview_image({}))

    def test_kinect_preview_is_converted_to_rgb_and_duplicated(self):
        server = self.make_server("none")
        server.enabled_camera_names = ("KINECT",)
        server.preview_camera = "KINECT"
        color_bgr = np.array([[[1, 2, 3]]], dtype=np.uint8)

        preview = server._build_preview_image(
            {"KINECT": {"color": color_bgr}}
        )

        expected_eye = np.array([[[3, 2, 1]]], dtype=np.uint8)
        np.testing.assert_array_equal(preview[:, :1], expected_eye)
        np.testing.assert_array_equal(preview[:, 1:], expected_eye)

    def test_handshake_describes_selected_camera_mode(self):
        server = self.make_server("none")
        server.img_shape = (720, 2560, 3)
        server.img_sock = Mock()

        server._send_handshake()

        metadata = server.img_sock.send_json.call_args.args[0]
        self.assertEqual(metadata["camera_mode"], "none")
        self.assertEqual(metadata["preview_camera"], "none")
        self.assertEqual(metadata["shape"], (720, 2560, 3))

    def test_zed_recording_keeps_both_eyes(self):
        server = self.make_server("zed")
        server._estimate_keypoint_distances = (
            lambda arm_q, hand_q, frames: np.zeros(15)
        )
        server.record_queue = Queue()
        server.arm_target_q_rad = np.zeros(7)
        server.hand_target_q = np.zeros(20)
        server.hand_command_q = np.zeros(20)
        server.last_valid_ee_pose = np.zeros(7)
        obs = {
            "arm_qpos": np.zeros(7),
            "hand_qpos": np.zeros(20),
            "ee_pose": np.zeros(7),
            "tactile": {},
            "images": {
                "ZED_M": {
                    "color": np.full((4, 6, 3), 10, dtype=np.uint8),
                    "color_right": np.full((4, 6, 3), 20, dtype=np.uint8),
                    "depth": np.zeros((4, 6), dtype=np.uint16),
                }
            },
        }

        server._queue_record_data(1.0, obs)

        images = server.record_queue.get_nowait()["observations"]["images"]
        self.assertEqual(set(images), {"ZED_M"})
        self.assertEqual(images["ZED_M"]["rgb"].shape, (2, 3, 3))
        self.assertEqual(images["ZED_M"]["rgb_right"].shape, (2, 3, 3))
        self.assertEqual(images["ZED_M"]["depth"].shape, (2, 3))

    def test_kinect_recording_includes_sync_pose_and_fixed_cloud(self):
        server = self.make_server("none")
        server._estimate_keypoint_distances = (
            lambda arm_q, hand_q, frames: np.zeros(15)
        )
        server.record_queue = Queue()
        server.arm_target_q_rad = np.zeros(7)
        server.hand_target_q = np.zeros(20)
        server.hand_command_q = np.zeros(20)
        server.last_valid_ee_pose = np.zeros(7)
        obs = {
            "arm_qpos": np.zeros(7),
            "hand_qpos": np.zeros(20),
            "ee_pose": np.zeros(7),
            "tactile": {},
            "images": {
                "KINECT": {
                    "color": np.zeros((4, 6, 3), dtype=np.uint8),
                    "depth": np.zeros((4, 6), dtype=np.uint16),
                    "valid": True,
                    "pose_valid": True,
                    "clouds_updated": True,
                    "frame_id": 8,
                    "capture_monotonic_ns": 100,
                    "capture_wall_ns": 200,
                    "receive_monotonic_ns": 110,
                    "sync_age_ms": 5.0,
                    "inference_ms": 4.0,
                    "intrinsics": np.eye(3, dtype=np.float32),
                    "object_pose_camera": np.eye(4, dtype=np.float32),
                    "pointcloud_xyz": np.zeros((16, 3), dtype=np.float32),
                    "pointcloud_rgb": np.zeros((16, 3), dtype=np.uint8),
                    "pointcloud_count": 3,
                }
            },
        }

        server._queue_record_data(1.0, obs)

        frame = server.record_queue.get_nowait()
        self.assertEqual(frame["observations"]["images"]["KINECT"]["rgb"].shape, (4, 6, 3))
        kinect = frame["observations"]["kinect"]
        self.assertEqual(kinect["frame_id"], 8)
        self.assertEqual(kinect["tracking_mode"], b"foundationpose")
        self.assertEqual(kinect["pointcloud_xyz"].shape, (16, 3))
        np.testing.assert_array_equal(kinect["object_pose_camera"], np.eye(4))


if __name__ == "__main__":
    unittest.main()
