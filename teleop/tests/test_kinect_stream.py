import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

import h5py
import numpy as np
import zmq

from utils.data_collector_wuji import DataCollectorWuji
from utils.kinect_stream import (
    KinectZmqReceiver,
    decode_kinect_message,
    encode_kinect_message,
    pack_fixed_pointcloud,
)


class KinectStreamCodecTest(unittest.TestCase):
    def test_round_trip_preserves_depth_geometry_pose_and_cloud(self):
        color = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
        depth = np.arange(24, dtype=np.uint16).reshape(4, 6) + 100
        intrinsics = np.array([[500, 0, 3], [0, 501, 2], [0, 0, 1]], np.float32)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, 3] = [0.1, 0.2, 0.3]
        points = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], np.float32)
        colors = np.array([[1, 2, 3], [4, 5, 6]], np.uint8)

        parts = encode_kinect_message(
            frame_id=7,
            color_bgr=color,
            depth_mm=depth,
            intrinsics=intrinsics,
            object_pose_camera=pose,
            capture_monotonic_ns=123,
            capture_wall_ns=456,
            inference_ms=4.5,
            pose_valid=True,
            clouds_updated=True,
            pointcloud_xyz=points,
            pointcloud_rgb=colors,
            jpeg_quality=100,
        )
        decoded = decode_kinect_message(parts)

        np.testing.assert_array_equal(decoded["depth"], depth)
        np.testing.assert_allclose(decoded["intrinsics"], intrinsics)
        np.testing.assert_allclose(decoded["object_pose_camera"], pose)
        np.testing.assert_allclose(decoded["pointcloud_xyz"], points)
        np.testing.assert_array_equal(decoded["pointcloud_rgb"], colors)
        self.assertEqual(decoded["color"].shape, color.shape)
        self.assertEqual(decoded["frame_id"], 7)
        self.assertTrue(decoded["clouds_updated"])

    def test_fixed_cloud_is_deterministically_limited_and_padded(self):
        points = np.arange(30, dtype=np.float32).reshape(10, 3)
        colors = np.arange(30, dtype=np.uint8).reshape(10, 3)

        xyz, rgb, count = pack_fixed_pointcloud(points, colors, 4)

        self.assertEqual(xyz.shape, (4, 3))
        self.assertEqual(rgb.shape, (4, 3))
        self.assertEqual(count, 4)
        np.testing.assert_array_equal(xyz[0], points[0])
        np.testing.assert_array_equal(xyz[-1], points[-1])

        xyz, rgb, count = pack_fixed_pointcloud(points[:2], colors[:2], 4)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(xyz[2:], 0)
        np.testing.assert_array_equal(rgb[2:], 0)

    def test_everything_sample_is_valid_without_an_object_pose(self):
        now = time.monotonic_ns()
        parts = encode_kinect_message(
            frame_id=8,
            color_bgr=np.zeros((4, 6, 3), np.uint8),
            depth_mm=np.ones((4, 6), np.uint16),
            intrinsics=np.eye(3, dtype=np.float32),
            object_pose_camera=np.eye(4, dtype=np.float32),
            capture_monotonic_ns=now,
            capture_wall_ns=time.time_ns(),
            inference_ms=3.0,
            pose_valid=False,
            sample_valid=True,
            segmentation_valid=True,
            tracking_mode="everything",
        )
        sample = decode_kinect_message(parts)
        receiver = KinectZmqReceiver.__new__(KinectZmqReceiver)
        receiver._lock = threading.Lock()
        receiver.max_age_ns = int(1e9)
        receiver._latest = sample

        frame = receiver.get_frames()["KINECT"]

        self.assertTrue(frame["valid"])
        self.assertTrue(frame["segmentation_valid"])
        self.assertFalse(frame["pose_valid"])
        self.assertEqual(frame["tracking_mode"], "everything")

    def test_receiver_marks_old_sample_invalid(self):
        receiver = KinectZmqReceiver.__new__(KinectZmqReceiver)
        receiver._lock = threading.Lock()
        receiver.max_age_ns = int(0.1 * 1e9)
        receiver._latest = {
            "capture_monotonic_ns": time.monotonic_ns() - int(0.5 * 1e9),
            "pose_valid": True,
        }

        frame = receiver.get_frames()["KINECT"]

        self.assertFalse(frame["valid"])
        self.assertGreater(frame["sync_age_ms"], 100.0)

    @unittest.skipUnless(
        os.environ.get("RUN_ZMQ_INTEGRATION_TESTS") == "1",
        "set RUN_ZMQ_INTEGRATION_TESTS=1 where local TCP sockets are allowed",
    )
    def test_tcp_receiver_keeps_latest_sample_and_fixes_cloud_shape(self):
        context = zmq.Context()
        publisher = context.socket(zmq.PUB)
        port = publisher.bind_to_random_port("tcp://127.0.0.1")
        receiver = KinectZmqReceiver(
            f"tcp://127.0.0.1:{port}",
            max_age_s=1.0,
            pointcloud_capacity=4,
        )
        try:
            time.sleep(0.1)  # Allow the PUB/SUB subscription to propagate.
            for frame_id in (1, 2):
                message = encode_kinect_message(
                    frame_id=frame_id,
                    color_bgr=np.zeros((4, 6, 3), np.uint8),
                    depth_mm=np.ones((4, 6), np.uint16),
                    intrinsics=np.eye(3, dtype=np.float32),
                    object_pose_camera=np.eye(4, dtype=np.float32),
                    capture_monotonic_ns=time.monotonic_ns(),
                    capture_wall_ns=time.time_ns(),
                    inference_ms=1.0,
                    pose_valid=True,
                    clouds_updated=True,
                    pointcloud_xyz=np.arange(18, dtype=np.float32).reshape(6, 3),
                    pointcloud_rgb=np.arange(18, dtype=np.uint8).reshape(6, 3),
                )
                publisher.send_multipart(message)
            deadline = time.monotonic() + 2.0
            frames = {}
            while time.monotonic() < deadline:
                frames = receiver.get_frames()
                if frames.get("KINECT", {}).get("frame_id") == 2:
                    break
                time.sleep(0.01)

            sample = frames["KINECT"]
            self.assertEqual(sample["frame_id"], 2)
            self.assertTrue(sample["valid"])
            self.assertEqual(sample["pointcloud_xyz"].shape, (4, 3))
            self.assertEqual(sample["pointcloud_count"], 4)
        finally:
            receiver.stop()
            publisher.close(0)
            context.term()


class StreamingCollectorTest(unittest.TestCase):
    def test_kinect_arrays_are_appended_compressed_and_atomically_finalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = DataCollectorWuji(
                temp_dir,
                hand_model="test-hand",
                hand_joint_names=tuple(f"joint_{i}" for i in range(20)),
                hand_urdf="right.urdf",
                flange_thickness=0.03,
                camera_mode="none",
                camera_names=("KINECT",),
                distance_keypoint_names=tuple(f"point_{i}" for i in range(15)),
                distance_T_base_camera=np.eye(4),
                flush_every_frames=1,
            )
            collector.start_recording("kinect_test")
            inprogress = collector.inprogress_filepath
            self.assertTrue(inprogress.is_file())

            frame = {
                "time": 1.0,
                "observations": {
                    "qpos": np.zeros(27),
                    "ee_pose": np.zeros(7),
                    "distances": np.zeros(15),
                    "tactile": {},
                    "images": {
                        "KINECT": {
                            "rgb": np.zeros((32, 48, 3), np.uint8),
                            "depth": np.zeros((32, 48), np.uint16),
                        }
                    },
                    "kinect": {
                        "valid": np.uint8(1),
                        "frame_id": np.int64(4),
                        "intrinsics": np.eye(3, dtype=np.float32),
                        "object_pose_camera": np.eye(4, dtype=np.float32),
                        "pointcloud_xyz": np.zeros((16, 3), np.float32),
                        "pointcloud_rgb": np.zeros((16, 3), np.uint8),
                        "pointcloud_count": np.int32(0),
                    },
                },
                "action": {"arm_target_q": np.zeros(7)},
            }
            collector.add_frame(frame)
            collector.add_frame(frame)
            collector.stop_recording()

            self.assertFalse(inprogress.exists())
            final_path = next(Path(temp_dir).glob("*.h5"))
            with h5py.File(final_path, "r") as h5_file:
                self.assertTrue(h5_file.attrs["recording_complete"])
                self.assertEqual(h5_file.attrs["writer"], "streaming-v1")
                self.assertEqual(
                    h5_file.attrs["distance_metric"], "nearest_euclidean"
                )
                self.assertEqual(
                    h5_file.attrs["distance_source"],
                    "kinect_observed_target_pointcloud",
                )
                np.testing.assert_array_equal(
                    h5_file.attrs["kinect_T_base_camera"], np.eye(4)
                )
                self.assertEqual(h5_file["observations/qpos"].shape, (2, 27))
                rgb = h5_file["observations/images/KINECT/rgb"]
                self.assertEqual(rgb.shape, (2, 32, 48, 3))
                self.assertEqual(rgb.compression, "lzf")
                self.assertEqual(
                    h5_file["observations/kinect/object_pose_camera"].shape,
                    (2, 4, 4),
                )

    def test_missing_and_late_fields_remain_aligned_with_the_timeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = DataCollectorWuji(
                temp_dir,
                hand_model="test-hand",
                hand_joint_names=("joint_0",),
                hand_urdf="right.urdf",
                flange_thickness=0.03,
                camera_mode="none",
                camera_names=(),
                distance_keypoint_names=(),
            )
            collector.start_recording("field_alignment")
            collector.add_frame(
                {
                    "time": np.float64(1.0),
                    "observations": {"qpos": np.array([1.0], np.float32)},
                }
            )
            collector.add_frame(
                {
                    "time": np.float64(2.0),
                    "observations": {
                        "qpos": np.array([2.0], np.float32),
                        "late_value": np.int32(7),
                    },
                }
            )
            collector.add_frame(
                {
                    "time": np.float64(3.0),
                    "observations": {"qpos": np.array([3.0], np.float32)},
                }
            )
            collector.stop_recording()

            final_path = next(Path(temp_dir).glob("*.h5"))
            with h5py.File(final_path, "r") as h5_file:
                np.testing.assert_array_equal(
                    h5_file["observations/late_value"][:],
                    np.array([0, 7, 0], dtype=np.int32),
                )
                self.assertEqual(h5_file["time"].shape, (3,))
                self.assertEqual(h5_file["observations/qpos"].shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
