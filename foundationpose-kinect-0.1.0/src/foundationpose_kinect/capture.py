"""Lossless RGB-D, calibration, and MeshLab-compatible point-cloud capture."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def dilate_binary_mask(mask, radius):
    """Dilate a 2D binary mask by an elliptical image-space radius."""
    binary = np.asarray(mask) > 0
    radius = int(radius)
    if binary.ndim != 2:
        raise ValueError("dilation mask must be two-dimensional")
    if radius < 0:
        raise ValueError("dilation radius cannot be negative")
    if radius == 0:
        return binary.copy()
    diameter = radius * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (diameter, diameter),
    )
    return cv2.dilate(binary.astype(np.uint8), kernel, iterations=1) > 0


def make_colored_pointcloud(
    color_bgr,
    depth_mm,
    depth_scale,
    intrinsics,
    *,
    mask=None,
):
    """Back-project valid aligned-depth pixels into the color camera frame."""
    color = np.asarray(color_bgr)
    depth = np.asarray(depth_mm)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    scale = float(depth_scale)

    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 uint8 image")
    if depth.dtype != np.uint16 or depth.shape != color.shape[:2]:
        raise ValueError("depth_mm must be uint16 and aligned with color_bgr")
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("intrinsics must be a finite 3x3 matrix")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or scale <= 0:
        raise ValueError("focal lengths and depth scale must be positive")

    selected = depth > 0
    if mask is not None:
        binary = np.asarray(mask) > 0
        if binary.shape != depth.shape:
            raise ValueError("point-cloud mask must match the aligned depth image")
        selected &= binary

    rows, columns = np.nonzero(selected)
    z = depth[selected].astype(np.float32) * np.float32(scale)
    x = (columns.astype(np.float32) - np.float32(matrix[0, 2])) * z
    x /= np.float32(matrix[0, 0])
    y = (rows.astype(np.float32) - np.float32(matrix[1, 2])) * z
    y /= np.float32(matrix[1, 1])
    points = np.column_stack((x, y, z)).astype(np.float32, copy=False)
    colors_rgb = np.ascontiguousarray(color[selected, ::-1])
    return points, colors_rgb


def write_colored_ply(path, points, colors_rgb):
    """Write XYZ float32 and RGB uint8 vertices as binary little-endian PLY."""
    destination = Path(path)
    xyz = np.asarray(points, dtype=np.float32)
    rgb = np.asarray(colors_rgb, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,):
        raise ValueError("PLY points must have shape Nx3")
    if rgb.shape != xyz.shape:
        raise ValueError("PLY RGB colors must have shape Nx3")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("PLY points must be finite")

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(xyz), dtype=vertex_dtype)
    vertices["x"], vertices["y"], vertices["z"] = xyz.T
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment coordinates: x right, y down, z forward, units meter\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)
    temporary.replace(destination)


def save_rgbd_capture(
    capture_dir,
    color_bgr,
    depth_mm,
    depth_scale,
    intrinsics,
    *,
    captured_at,
    mask=None,
    raw_mask=None,
    mask_dilation_px=0,
    save_scene_cloud=True,
    frame_id=None,
    factory_intrinsics=None,
    calibration=None,
    segmentation_valid=None,
):
    """Save synchronized images, optional scene/mask PLYs, and camera metadata."""
    output = Path(capture_dir)
    output.mkdir(parents=True, exist_ok=False)
    rgb_path = output / "rgb.png"
    depth_path = output / "depth_mm.png"
    scene_cloud_path = (
        output / "scene_cloud.ply" if save_scene_cloud else None
    )
    camera_path = output / "camera.json"

    color = np.asarray(color_bgr)
    depth = np.asarray(depth_mm)
    matrix = np.asarray(intrinsics, dtype=np.float64)
    if not cv2.imwrite(
        str(rgb_path),
        color,
        [cv2.IMWRITE_PNG_COMPRESSION, 3],
    ):
        raise RuntimeError(f"Failed to write RGB image: {rgb_path}")
    if not cv2.imwrite(
        str(depth_path),
        depth.astype(np.uint16, copy=False),
        [cv2.IMWRITE_PNG_COMPRESSION, 1],
    ):
        raise RuntimeError(f"Failed to write depth image: {depth_path}")

    scene_points = None
    if scene_cloud_path is not None:
        scene_points, scene_colors = make_colored_pointcloud(
            color,
            depth,
            depth_scale,
            matrix,
        )
        write_colored_ply(scene_cloud_path, scene_points, scene_colors)

    def save_mask_cloud(mask_value, mask_name, cloud_name):
        binary = np.asarray(np.asarray(mask_value) > 0, dtype=np.uint8)
        if binary.shape != depth.shape:
            raise ValueError("capture mask must match the aligned depth image")
        mask_path = output / mask_name
        cloud_path = output / cloud_name
        if not cv2.imwrite(str(mask_path), binary * 255):
            raise RuntimeError(f"Failed to write segmentation mask: {mask_path}")
        points, colors = make_colored_pointcloud(
            color,
            depth,
            depth_scale,
            matrix,
            mask=binary,
        )
        write_colored_ply(cloud_path, points, colors)
        return {
            "file": cloud_path.name,
            "mask_file": mask_path.name,
            "mask_pixels": int(np.count_nonzero(binary)),
            "points": len(points),
            "source": "nonzero aligned depth pixels inside the saved mask",
        }

    final_cloud = (
        None
        if mask is None
        else save_mask_cloud(mask, "mask.png", "masked_cloud.ply")
    )
    raw_cloud = None
    dilated_cloud = None
    dilation_radius = int(mask_dilation_px)
    if dilation_radius < 0:
        raise ValueError("mask dilation radius cannot be negative")
    if raw_mask is not None:
        raw_binary = np.asarray(raw_mask) > 0
        raw_cloud = save_mask_cloud(
            raw_binary,
            "mask_raw.png",
            "raw_mask_cloud.ply",
        )
        dilated_binary = dilate_binary_mask(raw_binary, dilation_radius)
        dilated_cloud = save_mask_cloud(
            dilated_binary,
            "mask_dilated.png",
            "dilated_mask_cloud.ply",
        )
        dilated_cloud["dilation_radius_px"] = dilation_radius
    elif dilation_radius:
        raise ValueError("mask dilation requires a raw mask")

    height, width = depth.shape
    valid_pixels = int(np.count_nonzero(depth))
    metadata = {
        "captured_at": str(captured_at),
        "frame_id": None if frame_id is None else int(frame_id),
        "camera": "Azure Kinect",
        "image_size": {"width": int(width), "height": int(height)},
        "camera_matrix": matrix.tolist(),
        "runtime_camera_matrix": matrix.tolist(),
        "factory_camera_matrix": (
            None
            if factory_intrinsics is None
            else np.asarray(factory_intrinsics, dtype=np.float64).tolist()
        ),
        "intrinsics": {
            "fx": float(matrix[0, 0]),
            "fy": float(matrix[1, 1]),
            "cx": float(matrix[0, 2]),
            "cy": float(matrix[1, 2]),
        },
        "depth": {
            "file": depth_path.name,
            "dtype": "uint16",
            "unit": "millimeter",
            "scale_m_per_unit": float(depth_scale),
            "invalid_value": 0,
            "aligned_to": "color",
            "valid_pixels": valid_pixels,
            "valid_ratio": valid_pixels / depth.size,
        },
        "color": {
            "file": rgb_path.name,
            "encoding": "PNG BGR8 decoded by OpenCV",
        },
        "scene_point_cloud": {
            "file": scene_cloud_path.name,
            "format": "binary_little_endian PLY 1.0",
            "points": len(scene_points),
            "source": "all nonzero pixels in the color-aligned depth image",
        }
        if scene_cloud_path is not None
        else None,
        "masked_point_cloud": (
            None
            if final_cloud is None
            else {
                **final_cloud,
                "mask_stage": "final_eroded",
                "segmentation_valid": (
                    None
                    if segmentation_valid is None
                    else bool(segmentation_valid)
                ),
            }
        ),
        "raw_mask_point_cloud": raw_cloud,
        "dilated_mask_point_cloud": dilated_cloud,
        "point_cloud_coordinates": {
            "units": "meter",
            "coordinate_frame": "color_camera",
            "axes": {"x": "right", "y": "down", "z": "forward"},
            "color_encoding": "RGB uint8",
            "hand_eye_transform_applied": False,
        },
        "calibration": calibration,
    }
    temporary_camera_path = camera_path.with_suffix(".json.tmp")
    temporary_camera_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    temporary_camera_path.replace(camera_path)
    return {
        "output": output,
        "scene_points": (
            None if scene_points is None else len(scene_points)
        ),
        "masked_points": (
            None if final_cloud is None else final_cloud["points"]
        ),
        "raw_mask_points": (
            None if raw_cloud is None else raw_cloud["points"]
        ),
        "dilated_mask_points": (
            None if dilated_cloud is None else dilated_cloud["points"]
        ),
    }
