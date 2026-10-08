"""Eye-to-hand calibration math for a fixed camera and robot-mounted target."""

from __future__ import annotations

import cv2
import numpy as np


HAND_EYE_METHODS = {
    "park": cv2.CALIB_HAND_EYE_PARK,
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _transform(rotation, translation):
    value = np.eye(4, dtype=np.float64)
    value[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    value[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return value


def validate_transform(value, label="transform"):
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{label} must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{label} rotation determinant must be +1")
    return transform


def rotation_angle_rad(rotation):
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    sine = np.linalg.norm(rotation - rotation.T) / (2.0 * np.sqrt(2.0))
    return float(np.arctan2(sine, cosine))


def mean_rotation(rotations):
    matrix = np.sum(np.asarray(rotations, dtype=np.float64), axis=0)
    left, _singular, right_t = np.linalg.svd(matrix)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right_t)
    return left @ correction @ right_t


def evaluate_eye_to_hand(T_base_camera, T_base_gripper, T_camera_target):
    """Measure how constant the inferred gripper-to-target mount remains."""
    camera = validate_transform(T_base_camera, "T_base_camera")
    if len(T_base_gripper) != len(T_camera_target) or len(T_base_gripper) < 3:
        raise ValueError("eye-to-hand evaluation requires at least three paired poses")

    mounts = []
    for index, (base_gripper, camera_target) in enumerate(
        zip(T_base_gripper, T_camera_target)
    ):
        base_gripper = validate_transform(
            base_gripper, f"T_base_gripper[{index}]"
        )
        camera_target = validate_transform(
            camera_target, f"T_camera_target[{index}]"
        )
        mounts.append(np.linalg.inv(base_gripper) @ camera @ camera_target)

    mean_mount = np.eye(4, dtype=np.float64)
    mean_mount[:3, :3] = mean_rotation([mount[:3, :3] for mount in mounts])
    mean_mount[:3, 3] = np.mean([mount[:3, 3] for mount in mounts], axis=0)
    translation_errors = np.asarray(
        [np.linalg.norm(mount[:3, 3] - mean_mount[:3, 3]) for mount in mounts]
    )
    rotation_errors = np.asarray(
        [
            rotation_angle_rad(mean_mount[:3, :3].T @ mount[:3, :3])
            for mount in mounts
        ]
    )
    translation_rms = float(np.sqrt(np.mean(translation_errors**2)))
    rotation_rms = float(np.sqrt(np.mean(rotation_errors**2)))
    return {
        "T_gripper_target_mean": mean_mount,
        "translation_errors_m": translation_errors,
        "rotation_errors_deg": np.rad2deg(rotation_errors),
        "translation_rms_m": translation_rms,
        "translation_max_m": float(np.max(translation_errors)),
        "rotation_rms_deg": float(np.rad2deg(rotation_rms)),
        "rotation_max_deg": float(np.max(np.rad2deg(rotation_errors))),
        # A 5 cm lever converts angular inconsistency into an approximate length.
        "score": translation_rms + 0.05 * rotation_rms,
    }


def solve_eye_to_hand(T_base_gripper, T_camera_target, methods=None):
    """Solve fixed-camera T_base_camera and rank OpenCV hand-eye methods."""
    if len(T_base_gripper) != len(T_camera_target) or len(T_base_gripper) < 3:
        raise ValueError("eye-to-hand calibration requires at least three pose pairs")
    base_gripper = [
        validate_transform(value, f"T_base_gripper[{index}]")
        for index, value in enumerate(T_base_gripper)
    ]
    camera_target = [
        validate_transform(value, f"T_camera_target[{index}]")
        for index, value in enumerate(T_camera_target)
    ]

    # For eye-to-hand, invert the robot poses before calibrateHandEye. OpenCV's
    # virtual "gripper" then represents base_link, so its camera-to-gripper
    # output is exactly T_base_camera.
    gripper_base = [np.linalg.inv(value) for value in base_gripper]
    method_names = tuple(methods or HAND_EYE_METHODS)
    results = {}
    for name in method_names:
        if name not in HAND_EYE_METHODS:
            raise ValueError(f"Unknown hand-eye method: {name}")
        try:
            rotation, translation = cv2.calibrateHandEye(
                [value[:3, :3] for value in gripper_base],
                [value[:3, 3] for value in gripper_base],
                [value[:3, :3] for value in camera_target],
                [value[:3, 3] for value in camera_target],
                method=HAND_EYE_METHODS[name],
            )
            transform = validate_transform(
                _transform(rotation, translation), f"{name} T_base_camera"
            )
            metrics = evaluate_eye_to_hand(
                transform, base_gripper, camera_target
            )
            results[name] = {"T_base_camera": transform, **metrics}
        except (cv2.error, ValueError, np.linalg.LinAlgError) as exc:
            results[name] = {"error": str(exc)}

    valid = {
        name: result for name, result in results.items() if "error" not in result
    }
    if not valid:
        raise RuntimeError(f"All hand-eye methods failed: {results}")
    best_method = min(valid, key=lambda name: valid[name]["score"])
    return best_method, results
