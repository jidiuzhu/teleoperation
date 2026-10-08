"""Kinect-to-robot hand-eye calibration loading and point transformation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HAND_EYE_CONVENTION = "T_base_camera"


def validate_T_base_camera(value):
    """Validate a rigid transform mapping Kinect camera points into base_link."""
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("T_base_camera must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("T_base_camera last row must be [0, 0, 0, 1]")

    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("T_base_camera rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("T_base_camera rotation determinant must be +1")
    return transform


def load_T_base_camera(path):
    """Load a completed hand-eye calibration JSON file."""
    calibration_path = Path(path).expanduser().resolve()
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Kinect hand-eye calibration not found: {calibration_path}")
    data = json.loads(calibration_path.read_text(encoding="utf-8"))
    if data.get("convention") != HAND_EYE_CONVENTION:
        raise ValueError(
            f"Kinect calibration convention must be {HAND_EYE_CONVENTION!r}"
        )
    if data.get("translation_units") != "meter":
        raise ValueError("Kinect calibration translation_units must be 'meter'")
    if data.get("calibrated") is not True:
        raise ValueError(
            "Kinect hand-eye calibration is still a placeholder; fill matrix "
            "and set calibrated=true"
        )
    return validate_T_base_camera(data.get("matrix"))


def camera_points_to_base(points_camera, T_base_camera):
    """Apply p_base = R_base_camera @ p_camera + t_base_camera to Nx3 points."""
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("Kinect camera points must have shape Nx3")
    if not np.all(np.isfinite(points)):
        raise ValueError("Kinect camera points contain non-finite values")
    transform = validate_T_base_camera(T_base_camera)
    return points @ transform[:3, :3].T + transform[:3, 3]


def nearest_point_distances(query_points, pointcloud):
    """Return one nearest Euclidean distance for every query point."""
    queries = np.asarray(query_points, dtype=np.float64)
    points = np.asarray(pointcloud, dtype=np.float64)
    if queries.ndim != 2 or queries.shape[1:] != (3,):
        raise ValueError("query_points must have shape Nx3")
    if points.ndim != 2 or points.shape[1:] != (3,) or len(points) == 0:
        raise ValueError("pointcloud must have non-empty shape Mx3")
    if not np.all(np.isfinite(queries)) or not np.all(np.isfinite(points)):
        raise ValueError("nearest-distance inputs contain non-finite values")

    distances = np.empty(len(queries), dtype=np.float64)
    for index, query in enumerate(queries):
        offsets = points - query
        squared = np.einsum("ij,ij->i", offsets, offsets)
        distances[index] = np.sqrt(np.min(squared))
    return distances


def keypoint_to_camera_cloud_distances(
    keypoints_base, pointcloud_camera, T_base_camera
):
    """Compute base-frame keypoint distances to a camera-frame point cloud."""
    points_base = camera_points_to_base(pointcloud_camera, T_base_camera)
    return nearest_point_distances(keypoints_base, points_base)
