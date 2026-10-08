import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.kinect_hand_eye import (
    camera_points_to_base,
    keypoint_to_camera_cloud_distances,
    load_T_base_camera,
    nearest_point_distances,
    validate_T_base_camera,
)


class KinectHandEyeTest(unittest.TestCase):
    def test_camera_points_are_transformed_with_T_base_camera(self):
        transform = np.array(
            [
                [0.0, -1.0, 0.0, 0.4],
                [1.0, 0.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        points_camera = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 0.5]])

        points_base = camera_points_to_base(points_camera, transform)

        np.testing.assert_allclose(
            points_base,
            np.array([[-1.6, 0.8, 3.3], [0.4, -1.2, 0.8]]),
        )

    def test_nearest_distance_returns_only_one_value_per_query(self):
        queries = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        cloud = np.array(
            [[0.0, 0.3, 0.4], [10.0, 0.0, 2.0], [50.0, 50.0, 50.0]]
        )

        distances = nearest_point_distances(queries, cloud)

        self.assertEqual(distances.shape, (2,))
        np.testing.assert_allclose(distances, [0.5, 2.0])

    def test_fifteen_base_keypoints_use_transformed_camera_cloud(self):
        transform = np.array(
            [
                [0.0, -1.0, 0.0, 0.4],
                [1.0, 0.0, 0.0, -0.2],
                [0.0, 0.0, 1.0, 0.3],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        keypoints_base = np.column_stack(
            (np.arange(15, dtype=np.float64), np.zeros((15, 2)))
        )
        expected = np.arange(1, 16, dtype=np.float64) * 1e-3
        cloud_base = keypoints_base + np.column_stack(
            (np.zeros(15), expected, np.zeros(15))
        )
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        cloud_camera = (cloud_base - translation) @ rotation

        distances = keypoint_to_camera_cloud_distances(
            keypoints_base, cloud_camera, transform
        )

        self.assertEqual(distances.shape, (15,))
        np.testing.assert_allclose(distances, expected, atol=1e-12)

    def test_placeholder_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "hand_eye.json"
            path.write_text(
                json.dumps(
                    {
                        "convention": "T_base_camera",
                        "translation_units": "meter",
                        "calibrated": False,
                        "matrix": np.eye(4).tolist(),
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "placeholder"):
                load_T_base_camera(path)

    def test_non_rigid_rotation_is_rejected(self):
        transform = np.eye(4)
        transform[0, 0] = 2.0

        with self.assertRaisesRegex(ValueError, "orthonormal"):
            validate_T_base_camera(transform)


if __name__ == "__main__":
    unittest.main()
