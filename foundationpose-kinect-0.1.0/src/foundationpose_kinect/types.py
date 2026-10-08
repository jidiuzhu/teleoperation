"""Typed public data contracts used by callers."""
from dataclasses import dataclass, field

import numpy as np


def _empty_points():
    return np.empty((0, 3), dtype=np.float32)


def _empty_colors():
    return np.empty((0, 3), dtype=np.uint8)


@dataclass(frozen=True)
class RGBDFrame:
    """One color-aligned RGB-D frame.

    ``color_bgr`` is HxWx3 uint8, ``depth_mm`` is HxW uint16 millimeters,
    and ``intrinsics`` is a 3x3 color-camera matrix.
    """

    color_bgr: np.ndarray
    depth_mm: np.ndarray
    intrinsics: np.ndarray
    frame_id: int | None = None

    def validated(self):
        color = np.asarray(self.color_bgr)
        depth = np.asarray(self.depth_mm)
        intrinsics = np.asarray(self.intrinsics)
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("color_bgr must be HxWx3 uint8")
        if depth.dtype != np.uint16 or depth.shape != color.shape[:2]:
            raise ValueError("depth_mm must be HxW uint16 aligned to color_bgr")
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError("intrinsics focal lengths must be positive")
        if not np.any(depth):
            raise ValueError("depth_mm contains no valid depth")
        return self


@dataclass(frozen=True)
class PointCloud:
    """Colored XYZ points expressed in ``coordinate_frame``."""

    points: np.ndarray = field(default_factory=_empty_points)
    colors_rgb: np.ndarray = field(default_factory=_empty_colors)
    coordinate_frame: str = "camera"

    def __post_init__(self):
        points = np.asarray(self.points)
        colors = np.asarray(self.colors_rgb)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("point cloud points must be Nx3")
        if colors.shape != points.shape:
            raise ValueError("point cloud colors must match points")

    def __len__(self):
        return len(self.points)


@dataclass(frozen=True)
class PointCloudConfig:
    stride: int = 3
    min_depth_m: float = 0.15
    max_depth_m: float = 2.0
    max_points: int = 120000
    mesh_points: int = 12000
    reference_frames: int = 5
    reference_voxel_m: float = 0.003
    crop_margin_m: float = 0.005
    mask_erosion_px: int = 0

    def __post_init__(self):
        if self.stride < 1 or self.max_points < 1 or self.mesh_points < 1:
            raise ValueError("point cloud counts and stride must be positive")
        if not 0 <= self.min_depth_m < self.max_depth_m:
            raise ValueError("invalid point cloud depth range")
        if self.reference_frames < 1 or self.reference_voxel_m <= 0:
            raise ValueError("invalid reference cloud configuration")
        if self.crop_margin_m < 0:
            raise ValueError("crop margin cannot be negative")
        if self.mask_erosion_px < 0:
            raise ValueError("mask erosion cannot be negative")


@dataclass(frozen=True)
class PoseResult:
    """One synchronous FoundationPose result and target-only point clouds."""

    frame_id: int
    pose: np.ndarray
    inference_ms: float
    observed: PointCloud
    mesh: PointCloud
    reference: PointCloud
    initialized: bool
    clouds_updated: bool = True

    def __post_init__(self):
        pose = np.asarray(self.pose)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("pose must be a finite 4x4 matrix")

    @property
    def translation_m(self):
        return np.asarray(self.pose[:3, 3], dtype=np.float32)
