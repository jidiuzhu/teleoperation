"""Shared multimodal observation construction for Wuji ACTBH training and deployment."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.kinect_hand_eye import (
    keypoint_to_camera_cloud_distances,
    validate_T_base_camera,
)


RGB_KEY = "observation.images.KINECT_rgb"
DEPTH_IMAGE_KEY = "observation.images.KINECT_depth"
STATE_KEY = "observation.state"
ENVIRONMENT_STATE_KEY = "observation.environment_state"
DISTANCE_KEY = "observation.hand_object_distances"
POINT_CLOUD_KEY = "observation.point_cloud_global"
ACTION_KEY = "action"


@dataclass(frozen=True)
class ACTBHObservationConfig:
    image_height: int = 240
    image_width: int = 320
    point_count: int = 1024
    depth_min_m: float = 0.15
    depth_max_m: float = 1.50

    def validate(self) -> None:
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")
        if not 0 <= self.depth_min_m < self.depth_max_m:
            raise ValueError("depth range must satisfy 0 <= min < max")


def require_rigid_transform(value: np.ndarray, *, label: str) -> np.ndarray:
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{label} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
        raise ValueError(f"{label} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError(f"{label} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3):
        raise ValueError(f"{label} rotation determinant is not +1")
    return transform


def resize_rgb(rgb_hwc: np.ndarray, height: int, width: int) -> torch.Tensor:
    rgb = np.asarray(rgb_hwc)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"RGB frame must have shape HxWx3, got {rgb.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float()
    resized = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return resized.squeeze(0).round().clamp(0, 255).to(torch.uint8)


def depth_to_resnet_image(
    depth_mm: np.ndarray,
    height: int,
    width: int,
    depth_min_m: float,
    depth_max_m: float,
) -> torch.Tensor:
    """Encode metric depth as a three-channel uint8 image for the ACT ResNet."""
    depth = np.asarray(depth_mm)
    if depth.ndim != 2:
        raise ValueError(f"Depth frame must have shape HxW, got {depth.shape}")

    depth_m = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0).unsqueeze(0).float() * 0.001
    resized = F.interpolate(depth_m, size=(height, width), mode="nearest").squeeze(0).squeeze(0)
    valid = torch.isfinite(resized) & (resized > 0)
    normalized = (resized.clamp(depth_min_m, depth_max_m) - depth_min_m) / (
        depth_max_m - depth_min_m
    )
    normalized = torch.where(valid, normalized, torch.zeros_like(normalized))
    encoded = (normalized * 255.0).round().clamp(0, 255).to(torch.uint8)
    return encoded.unsqueeze(0).repeat(3, 1, 1)


def resample_and_transform_point_cloud(
    points_camera: np.ndarray,
    source_count: int,
    target_count: int,
    T_base_camera: np.ndarray,
) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud must have shape Nx3, got {points.shape}")
    if source_count <= 0 or source_count > len(points):
        raise ValueError(f"Invalid pointcloud_count={source_count} for shape {points.shape}")

    valid_points = points[:source_count]
    if not np.isfinite(valid_points).all():
        raise ValueError("Point cloud contains NaN or infinity in its valid prefix")

    if source_count >= target_count:
        indices = np.linspace(0, source_count - 1, target_count, dtype=np.int64)
    else:
        indices = np.arange(target_count, dtype=np.int64) % source_count
    sampled = valid_points[indices].astype(np.float64, copy=False)

    transform = require_rigid_transform(T_base_camera, label="T_base_camera")
    points_base = sampled @ transform[:3, :3].T + transform[:3, 3]
    return points_base.astype(np.float32)


def assemble_actbh_observation(
    *,
    rgb_hwc: np.ndarray,
    depth_mm: np.ndarray,
    arm_q_rad: np.ndarray,
    hand_q_rad: np.ndarray,
    distances_m: np.ndarray,
    pointcloud_camera: np.ndarray,
    pointcloud_count: int,
    T_base_camera: np.ndarray,
    config: ACTBHObservationConfig = ACTBHObservationConfig(),
) -> dict[str, np.ndarray]:
    """Build one unbatched observation in the format expected by ``predict_action``."""
    config.validate()
    arm_q = np.asarray(arm_q_rad, dtype=np.float32)
    hand_q = np.asarray(hand_q_rad, dtype=np.float32)
    distances = np.asarray(distances_m, dtype=np.float32)
    if arm_q.shape != (7,) or hand_q.shape != (20,):
        raise ValueError(f"Expected arm/hand shapes (7,)/(20,), got {arm_q.shape}/{hand_q.shape}")
    if distances.shape != (15,):
        raise ValueError(f"Expected 15 hand-object distances, got {distances.shape}")
    if not all(np.isfinite(value).all() for value in (arm_q, hand_q, distances)):
        raise ValueError("ACTBH state or distance input contains NaN or infinity")

    base_camera = require_rigid_transform(T_base_camera, label="T_base_camera")
    point_cloud_base = resample_and_transform_point_cloud(
        pointcloud_camera,
        source_count=int(pointcloud_count),
        target_count=config.point_count,
        T_base_camera=base_camera,
    )
    rgb_chw = resize_rgb(rgb_hwc, config.image_height, config.image_width)
    depth_chw = depth_to_resnet_image(
        depth_mm,
        config.image_height,
        config.image_width,
        config.depth_min_m,
        config.depth_max_m,
    )
    return {
        RGB_KEY: rgb_chw.permute(1, 2, 0).contiguous().numpy(),
        DEPTH_IMAGE_KEY: depth_chw.permute(1, 2, 0).contiguous().numpy(),
        STATE_KEY: np.concatenate([arm_q, hand_q]).astype(np.float32),
        ENVIRONMENT_STATE_KEY: distances,
        POINT_CLOUD_KEY: point_cloud_base,
    }


class LiveACTBHObservationBuilder:
    """Construct model observations from live Kinect and measured robot state."""

    def __init__(
        self,
        *,
        T_base_camera: np.ndarray,
        keypoint_kinematics,
        keypoint_specs,
        flange_thickness_m: float,
        config: ACTBHObservationConfig = ACTBHObservationConfig(),
    ):
        self.T_base_camera = validate_T_base_camera(T_base_camera)
        self.keypoint_kinematics = keypoint_kinematics
        self.keypoint_specs = tuple(keypoint_specs)
        if len(self.keypoint_specs) != 15:
            raise ValueError(f"ACTBH requires 15 distance keypoints, got {len(self.keypoint_specs)}")
        self.keypoint_names = tuple(str(spec["name"]) for spec in self.keypoint_specs)
        self.flange_thickness_m = float(flange_thickness_m)
        if not np.isfinite(self.flange_thickness_m) or self.flange_thickness_m < 0:
            raise ValueError("flange_thickness_m must be finite and non-negative")
        config.validate()
        self.config = config

    def build(self, arm_q_rad: np.ndarray, hand_q_rad: np.ndarray, kinect_frame: dict) -> dict[str, np.ndarray]:
        if not isinstance(kinect_frame, dict) or not kinect_frame.get("valid", False):
            raise ValueError("Kinect frame is missing, stale, or invalid")

        pointcloud = np.asarray(kinect_frame.get("pointcloud_xyz"), dtype=np.float32)
        pointcloud_count = int(kinect_frame.get("pointcloud_count", 0))
        positions = self.keypoint_kinematics.estimate_keypoints(
            arm_q_rad,
            hand_q_rad,
            self.flange_thickness_m,
            keypoint_specs=self.keypoint_specs,
            arm_angle_unit="radian",
        )
        keypoints_base = np.asarray(
            [positions[name] for name in self.keypoint_names],
            dtype=np.float64,
        )
        if pointcloud.ndim != 2 or pointcloud.shape[1:] != (3,):
            raise ValueError(f"Kinect point cloud must have shape Nx3, got {pointcloud.shape}")
        if pointcloud_count <= 0 or pointcloud_count > len(pointcloud):
            raise ValueError(
                f"Invalid Kinect pointcloud_count={pointcloud_count} for {pointcloud.shape}"
            )
        distances = keypoint_to_camera_cloud_distances(
            keypoints_base,
            pointcloud[:pointcloud_count],
            self.T_base_camera,
        )
        color_rgb = cv2.cvtColor(np.asarray(kinect_frame["color"]), cv2.COLOR_BGR2RGB)
        return assemble_actbh_observation(
            rgb_hwc=color_rgb,
            depth_mm=kinect_frame["depth"],
            arm_q_rad=arm_q_rad,
            hand_q_rad=hand_q_rad,
            distances_m=distances,
            pointcloud_camera=pointcloud,
            pointcloud_count=pointcloud_count,
            T_base_camera=self.T_base_camera,
            config=self.config,
        )
