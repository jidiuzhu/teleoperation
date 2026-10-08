"""Vectorized target point-cloud operations kept outside the pose fast path."""
from pathlib import Path

import numpy as np

from .types import PointCloud, PointCloudConfig, RGBDFrame


def empty_cloud():
    return PointCloud()


def load_obj_surface(path, count):
    vertices = []
    faces = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                indices = []
                for token in line.split()[1:]:
                    index = int(token.split("/", 1)[0])
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                for offset in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[offset], indices[offset + 1]))
    points = np.asarray(vertices, dtype=np.float32)
    if not len(points):
        raise ValueError(f"mesh contains no vertices: {path}")
    bounds = points.min(axis=0), points.max(axis=0)
    if not faces:
        indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
        return bounds, points[indices]
    triangles = points[np.asarray(faces, dtype=np.int32)]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1) * 0.5
    valid = areas > 1e-12
    triangles, areas = triangles[valid], areas[valid]
    if not len(triangles):
        raise ValueError(f"mesh contains no non-degenerate faces: {path}")
    generator = np.random.default_rng(0)
    selected = generator.choice(len(triangles), count, p=areas / areas.sum())
    u, v = generator.random((2, count))
    reflected = u + v > 1
    u[reflected], v[reflected] = 1 - u[reflected], 1 - v[reflected]
    sampled = triangles[selected]
    surface = (sampled[:, 0] + u[:, None] * (sampled[:, 1] - sampled[:, 0]) +
               v[:, None] * (sampled[:, 2] - sampled[:, 0])).astype(np.float32)
    return bounds, surface


def erode_binary_mask(mask, radius):
    """Erode a 2D mask by a circular radius without extra dependencies."""
    source = np.asarray(mask, dtype=bool)
    radius = int(radius)
    if source.ndim != 2:
        raise ValueError("erosion mask must be two-dimensional")
    if radius < 0:
        raise ValueError("erosion radius cannot be negative")
    if radius == 0:
        return source.copy()

    height, width = source.shape
    padded = np.pad(source, radius, mode="constant", constant_values=False)
    eroded = np.ones_like(source)
    diameter = radius * 2 + 1
    for row_offset in range(diameter):
        for column_offset in range(diameter):
            row_delta = row_offset - radius
            column_delta = column_offset - radius
            if row_delta * row_delta + column_delta * column_delta > radius * radius:
                continue
            eroded &= padded[
                row_offset:row_offset + height,
                column_offset:column_offset + width,
            ]
    return eroded


def observed_cloud(frame: RGBDFrame, pose, bounds, config: PointCloudConfig, mask=None):
    stride = config.stride
    depth = frame.depth_mm[::stride, ::stride].astype(np.float32) * 0.001
    rows, columns = np.indices(depth.shape, dtype=np.float32)
    u, v = columns * stride, rows * stride
    valid = ((depth >= config.min_depth_m) & (depth <= config.max_depth_m) &
             np.isfinite(depth))
    if mask is not None:
        valid &= np.asarray(mask)[::stride, ::stride] > 0
    intrinsics = np.asarray(frame.intrinsics)
    z = depth[valid]
    points = np.column_stack((
        (u[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
        (v[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1], z)).astype(np.float32)
    colors = frame.color_bgr[::stride, ::stride, ::-1][valid].astype(np.uint8)
    if len(points):
        rotation = np.asarray(pose[:3, :3], dtype=np.float32)
        translation = np.asarray(pose[:3, 3], dtype=np.float32)
        local = (points - translation) @ rotation
        low, high = bounds
        margin = config.crop_margin_m
        keep = ((local >= low - margin) & (local <= high + margin)).all(axis=1)
        target_mask = np.zeros_like(valid)
        target_mask[valid] = keep
        if config.mask_erosion_px:
            erosion_cells = max(
                1,
                int(np.ceil(config.mask_erosion_px / stride)),
            )
            target_mask = erode_binary_mask(target_mask, erosion_cells)
        keep = target_mask[valid]
        points, colors = points[keep], colors[keep]
    if len(points) > config.max_points:
        selected = np.linspace(0, len(points) - 1, config.max_points, dtype=np.int64)
        points, colors = points[selected], colors[selected]
    return PointCloud(points, colors, "camera")


def masked_cloud(frame: RGBDFrame, mask, config: PointCloudConfig):
    """Back-project valid depth pixels selected by a model-free image mask."""
    frame = frame.validated()
    target_mask = np.asarray(mask) > 0
    if target_mask.shape != frame.depth_mm.shape:
        raise ValueError("point-cloud mask must match the RGB-D frame dimensions")
    if config.mask_erosion_px:
        target_mask = erode_binary_mask(target_mask, config.mask_erosion_px)

    stride = config.stride
    depth = frame.depth_mm[::stride, ::stride].astype(np.float32) * 0.001
    selected_mask = target_mask[::stride, ::stride]
    rows, columns = np.indices(depth.shape, dtype=np.float32)
    u, v = columns * stride, rows * stride
    valid = (
        selected_mask
        & (depth >= config.min_depth_m)
        & (depth <= config.max_depth_m)
        & np.isfinite(depth)
    )
    intrinsics = np.asarray(frame.intrinsics, dtype=np.float32)
    z = depth[valid]
    points = np.column_stack(
        (
            (u[valid] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (v[valid] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    ).astype(np.float32)
    colors = frame.color_bgr[::stride, ::stride, ::-1][valid].astype(np.uint8)
    if len(points) > config.max_points:
        indices = np.linspace(0, len(points) - 1, config.max_points, dtype=np.int64)
        points, colors = points[indices], colors[indices]
    return PointCloud(points, colors, "camera")


def transform_cloud(model_points, pose, color):
    rotation = np.asarray(pose[:3, :3], dtype=np.float32)
    translation = np.asarray(pose[:3, 3], dtype=np.float32)
    points = np.asarray(model_points, dtype=np.float32) @ rotation.T + translation
    colors = np.broadcast_to(np.asarray(color, dtype=np.uint8), points.shape).copy()
    return PointCloud(points, colors, "camera")


class ReferenceAccumulator:
    def __init__(self, frames, voxel):
        self.target_frames = frames
        self.voxel = voxel
        self.frames = 0
        self._points = []
        self._colors = []
        self.model_points = None
        self.model_colors = None

    @property
    def ready(self):
        return self.model_points is not None

    def add(self, cloud, pose):
        if self.ready or not len(cloud):
            return
        rotation = np.asarray(pose[:3, :3], dtype=np.float32)
        translation = np.asarray(pose[:3, 3], dtype=np.float32)
        self._points.append((cloud.points - translation) @ rotation)
        self._colors.append(cloud.colors_rgb)
        self.frames += 1
        if self.frames >= self.target_frames:
            self._fuse()

    def _fuse(self):
        points = np.concatenate(self._points)
        colors = np.concatenate(self._colors).astype(np.float64)
        voxels = np.floor(points / self.voxel).astype(np.int64)
        _, inverse = np.unique(voxels, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.zeros((len(counts), 3), dtype=np.float64)
        color_sums = np.zeros_like(sums)
        np.add.at(sums, inverse, points)
        np.add.at(color_sums, inverse, colors)
        self.model_points = (sums / counts[:, None]).astype(np.float32)
        self.model_colors = np.clip(color_sums / counts[:, None], 0, 255).astype(np.uint8)
        self._points = self._colors = None

    def at_pose(self, pose):
        if self.ready:
            points, colors = self.model_points, self.model_colors
        elif self._points:
            points, colors = np.concatenate(self._points), np.concatenate(self._colors)
        else:
            return empty_cloud()
        rotation = np.asarray(pose[:3, :3], dtype=np.float32)
        translation = np.asarray(pose[:3, 3], dtype=np.float32)
        return PointCloud(points @ rotation.T + translation, colors, "camera")
