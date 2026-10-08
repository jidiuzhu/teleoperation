import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from scripts.teleop_real_server_wuji import TeleopServer
from utils.data_collector_wuji import DataCollectorWuji
from utils.kinect_hand_eye import load_T_base_camera, nearest_point_distances
from utils.wuji_keypoint_visualization import DEFAULT_KEYPOINT_SPECS


class WujiDistanceCollectionTest(unittest.TestCase):
    def make_server(self):
        server = TeleopServer.__new__(TeleopServer)
        server.flange_thickness = 0.035
        server.distance_keypoint_specs = DEFAULT_KEYPOINT_SPECS
        server.distance_keypoint_names = tuple(
            spec["name"] for spec in DEFAULT_KEYPOINT_SPECS
        )
        keypoints = {
            name: np.array([float(index), 0.0, 0.0])
            for index, name in enumerate(server.distance_keypoint_names)
        }

        class FixedKeypointKinematics:
            def estimate_keypoints(self, *_args, **_kwargs):
                return keypoints

        server.keypoint_kinematics = FixedKeypointKinematics()
        server.T_base_camera = np.array(
            [
                [0.0, -1.0, 0.0, 0.4],
                [1.0, 0.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        return server

    def test_distances_use_hand_eye_transform_and_nearest_cloud_point(self):
        server = self.make_server()
        positions = server.keypoint_kinematics.estimate_keypoints(
            None
        )
        expected = np.arange(1, 16, dtype=np.float64) * 1e-3
        points_base = np.asarray(
            [
                positions[name] + np.array([0.0, distance, 0.0])
                for name, distance in zip(server.distance_keypoint_names, expected)
            ]
        )
        rotation = server.T_base_camera[:3, :3]
        translation = server.T_base_camera[:3, 3]
        points_camera = (points_base - translation) @ rotation
        frame = {
            "valid": True,
            "pointcloud_xyz": points_camera.astype(np.float32),
            "pointcloud_count": len(points_camera),
        }

        distances = server._estimate_keypoint_distances(
            np.zeros(7), np.zeros(20), {"KINECT": frame}
        )

        self.assertEqual(distances.shape, (15,))
        np.testing.assert_allclose(distances, expected, atol=1e-6)

    def test_invalid_or_empty_cloud_produces_fifteen_nan_values(self):
        server = self.make_server()

        distances = server._estimate_keypoint_distances(
            np.zeros(7),
            np.zeros(20),
            {
                "KINECT": {
                    "valid": False,
                    "pointcloud_xyz": np.zeros((4, 3)),
                    "pointcloud_count": 4,
                }
            },
        )

        self.assertEqual(distances.shape, (15,))
        self.assertTrue(np.all(np.isnan(distances)))

    def test_calibration_file_rejects_placeholder_and_loads_rigid_transform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            calibration_path = Path(temp_dir) / "hand_eye.json"
            calibration = {
                "convention": "T_base_camera",
                "translation_units": "meter",
                "calibrated": False,
                "matrix": np.eye(4).tolist(),
            }
            calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "placeholder"):
                load_T_base_camera(calibration_path)

            calibration["calibrated"] = True
            calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
            np.testing.assert_array_equal(
                load_T_base_camera(calibration_path), np.eye(4)
            )

    def test_nearest_point_distances_returns_one_value_per_query(self):
        queries = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        cloud = np.array([[0.0, 0.3, 0.4], [10.0, 0.0, 2.0]])

        np.testing.assert_allclose(
            nearest_point_distances(queries, cloud), [0.5, 2.0]
        )

    def test_hdf5_contains_distances_and_reproducibility_metadata(self):
        names = tuple(spec["name"] for spec in DEFAULT_KEYPOINT_SPECS)
        T_base_camera = np.array(
            [
                [1.0, 0.0, 0.0, 0.40],
                [0.0, 1.0, 0.0, 0.00],
                [0.0, 0.0, 1.0, 0.30],
                [0.0, 0.0, 0.0, 1.00],
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            collector = DataCollectorWuji(
                temp_dir,
                hand_model="test-hand",
                hand_joint_names=tuple(f"joint_{i}" for i in range(20)),
                hand_urdf="right.urdf",
                flange_thickness=0.035,
                camera_mode="none",
                camera_names=("KINECT",),
                distance_keypoint_names=names,
                distance_T_base_camera=T_base_camera,
            )
            collector.start_recording("distance_test")
            collector.add_frame(
                {
                    "time": 1.0,
                    "observations": {
                        "qpos": np.zeros(27),
                        "ee_pose": np.zeros(7),
                        "distances": np.arange(15, dtype=np.float64) / 100.0,
                        "images": {},
                        "tactile": {},
                    },
                    "action": {
                        "arm_target_q": np.zeros(7),
                        "hand_target_q": np.zeros(20),
                        "hand_command_q": np.zeros(20),
                        "ee_target_pose": np.zeros(7),
                    },
                }
            )
            collector.stop_recording()

            h5_path = next(Path(temp_dir).glob("*.h5"))
            with h5py.File(h5_path, "r") as h5_file:
                self.assertEqual(h5_file["observations/distances"].shape, (1, 15))
                self.assertEqual(h5_file.attrs["distance_units"], "meter")
                self.assertEqual(
                    h5_file.attrs["distance_coordinate_frame"], "base_link"
                )
                self.assertEqual(
                    tuple(json.loads(h5_file.attrs["distance_keypoint_names"])),
                    names,
                )
                self.assertEqual(
                    h5_file.attrs["distance_metric"], "nearest_euclidean"
                )
                self.assertEqual(
                    h5_file.attrs["distance_source"],
                    "kinect_observed_target_pointcloud",
                )
                np.testing.assert_array_equal(
                    h5_file.attrs["kinect_T_base_camera"], T_base_camera
                )


if __name__ == "__main__":
    unittest.main()
