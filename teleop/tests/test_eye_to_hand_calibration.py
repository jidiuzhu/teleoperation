import unittest

import numpy as np

from utils.eye_to_hand_calibration import (
    evaluate_eye_to_hand,
    solve_eye_to_hand,
)


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cosine = np.cos(angle)
    sine = np.sin(angle)
    one_minus = 1.0 - cosine
    return np.array(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ]
    )


def transform(rotation, translation):
    value = np.eye(4)
    value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


class EyeToHandCalibrationTest(unittest.TestCase):
    def test_synthetic_eye_to_hand_recovers_T_base_camera(self):
        expected = transform(axis_angle([0, 0, 1], 0.4), [0.5, -0.2, 0.8])
        gripper_target = transform(
            axis_angle([1, 0, 0], -0.3), [0.05, 0.02, 0.1]
        )
        generator = np.random.default_rng(3)
        T_base_gripper = []
        T_camera_target = []
        for _index in range(24):
            axis = generator.normal(size=3)
            axis /= np.linalg.norm(axis)
            base_gripper = transform(
                axis_angle(axis, generator.uniform(-1.0, 1.0)),
                generator.uniform([0.2, -0.4, 0.2], [0.8, 0.4, 0.8]),
            )
            T_base_gripper.append(base_gripper)
            T_camera_target.append(
                np.linalg.inv(expected) @ base_gripper @ gripper_target
            )

        best_method, results = solve_eye_to_hand(
            T_base_gripper, T_camera_target
        )

        self.assertIn(best_method, results)
        np.testing.assert_allclose(
            results[best_method]["T_base_camera"], expected, atol=1e-8
        )
        self.assertLess(results[best_method]["translation_rms_m"], 1e-10)
        self.assertLess(results[best_method]["rotation_rms_deg"], 1e-8)

    def test_mount_consistency_exposes_wrong_extrinsic(self):
        base_gripper = [
            transform(np.eye(3), [0.1 * index, 0.0, 0.0])
            for index in range(3)
        ]
        camera_target = [value.copy() for value in base_gripper]
        wrong_camera = transform(np.eye(3), [0.0, 0.1, 0.0])

        metrics = evaluate_eye_to_hand(
            wrong_camera, base_gripper, camera_target
        )

        self.assertGreater(metrics["translation_rms_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
