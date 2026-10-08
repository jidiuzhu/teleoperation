"""Versioned ZMQ transport for synchronized Azure Kinect samples."""

from __future__ import annotations

import json
import threading
import time

import cv2
import numpy as np
import zmq


KINECT_TOPIC = b"teleop.kinect.v1"
KINECT_SCHEMA = "teleop.kinect.v1"


def _as_array(value, dtype, shape, label):
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must have finite shape {shape}, got {array.shape}")
    return array


def _encode_image(extension, image, parameters):
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode Kinect image as {extension}")
    return encoded.tobytes()


def encode_kinect_message(
    *,
    frame_id,
    color_bgr,
    depth_mm,
    intrinsics,
    object_pose_camera,
    capture_monotonic_ns,
    capture_wall_ns,
    inference_ms,
    pose_valid,
    sample_valid=None,
    segmentation_valid=False,
    tracking_mode="foundationpose",
    clouds_updated=False,
    pointcloud_xyz=None,
    pointcloud_rgb=None,
    source_id="azure-kinect",
    jpeg_quality=90,
    depth_png_compression=1,
):
    """Encode one sample as topic, JSON metadata, RGB, depth, XYZ and color parts."""
    color = np.ascontiguousarray(color_bgr, dtype=np.uint8)
    depth = np.ascontiguousarray(depth_mm, dtype=np.uint16)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("Kinect color_bgr must be HxWx3 uint8")
    if depth.shape != color.shape[:2]:
        raise ValueError("Kinect depth_mm must be aligned to color_bgr")

    matrix = _as_array(intrinsics, np.float32, (3, 3), "Kinect intrinsics")
    pose = _as_array(
        object_pose_camera,
        np.float32,
        (4, 4),
        "Kinect object_pose_camera",
    )

    points = np.asarray(
        np.empty((0, 3), dtype=np.float32)
        if pointcloud_xyz is None
        else pointcloud_xyz,
        dtype=np.float32,
    )
    colors = np.asarray(
        np.empty((0, 3), dtype=np.uint8)
        if pointcloud_rgb is None
        else pointcloud_rgb,
        dtype=np.uint8,
    )
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("Kinect pointcloud_xyz must have shape Nx3")
    if colors.shape != points.shape:
        raise ValueError("Kinect pointcloud_rgb must match pointcloud_xyz")
    if not np.all(np.isfinite(points)):
        raise ValueError("Kinect pointcloud_xyz contains non-finite values")
    points = np.ascontiguousarray(points)
    colors = np.ascontiguousarray(colors)
    if tracking_mode not in ("foundationpose", "everything"):
        raise ValueError(f"Unsupported Kinect tracking_mode: {tracking_mode!r}")
    if sample_valid is None:
        sample_valid = pose_valid

    metadata = {
        "schema": KINECT_SCHEMA,
        "source_id": str(source_id),
        "frame_id": int(frame_id),
        "capture_monotonic_ns": int(capture_monotonic_ns),
        "capture_wall_ns": int(capture_wall_ns),
        "publish_monotonic_ns": time.monotonic_ns(),
        "color_shape": list(color.shape),
        "depth_shape": list(depth.shape),
        "intrinsics": matrix.reshape(-1).tolist(),
        "object_pose_camera": pose.reshape(-1).tolist(),
        "pose_convention": "T_camera_object",
        "depth_units": "millimeter",
        "pointcloud_coordinate_frame": "camera",
        "pointcloud_count": int(len(points)),
        "inference_ms": float(inference_ms),
        "tracking_mode": tracking_mode,
        "sample_valid": bool(sample_valid),
        "pose_valid": bool(pose_valid),
        "segmentation_valid": bool(segmentation_valid),
        "clouds_updated": bool(clouds_updated and len(points)),
    }
    color_bytes = _encode_image(
        ".jpg",
        color,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    depth_bytes = _encode_image(
        ".png",
        depth,
        [int(cv2.IMWRITE_PNG_COMPRESSION), int(depth_png_compression)],
    )
    return (
        KINECT_TOPIC,
        json.dumps(metadata, separators=(",", ":")).encode("utf-8"),
        color_bytes,
        depth_bytes,
        points.tobytes(),
        colors.tobytes(),
    )


def decode_kinect_message(parts):
    """Decode and validate one multipart Kinect message."""
    if len(parts) != 6 or parts[0] != KINECT_TOPIC:
        raise ValueError("Unexpected Kinect multipart message")
    metadata = json.loads(parts[1].decode("utf-8"))
    if metadata.get("schema") != KINECT_SCHEMA:
        raise ValueError(f"Unsupported Kinect schema: {metadata.get('schema')!r}")

    color = cv2.imdecode(np.frombuffer(parts[2], np.uint8), cv2.IMREAD_COLOR)
    depth = cv2.imdecode(np.frombuffer(parts[3], np.uint8), cv2.IMREAD_UNCHANGED)
    color_shape = tuple(metadata["color_shape"])
    depth_shape = tuple(metadata["depth_shape"])
    if color is None or color.shape != color_shape or color.dtype != np.uint8:
        raise ValueError("Decoded Kinect color does not match its metadata")
    if depth is None or depth.shape != depth_shape or depth.dtype != np.uint16:
        raise ValueError("Decoded Kinect depth does not match its metadata")
    if depth.shape != color.shape[:2]:
        raise ValueError("Decoded Kinect RGB and depth are not aligned")

    point_count = int(metadata["pointcloud_count"])
    if point_count < 0:
        raise ValueError("Kinect pointcloud_count cannot be negative")
    expected_xyz_bytes = point_count * 3 * np.dtype(np.float32).itemsize
    expected_rgb_bytes = point_count * 3 * np.dtype(np.uint8).itemsize
    if len(parts[4]) != expected_xyz_bytes or len(parts[5]) != expected_rgb_bytes:
        raise ValueError("Kinect point-cloud payload size does not match metadata")

    points = np.frombuffer(parts[4], dtype=np.float32).reshape(point_count, 3).copy()
    colors = np.frombuffer(parts[5], dtype=np.uint8).reshape(point_count, 3).copy()
    if not np.all(np.isfinite(points)):
        raise ValueError("Decoded Kinect point cloud contains non-finite values")

    return {
        "color": color,
        "depth": depth,
        "intrinsics": _as_array(
            metadata["intrinsics"], np.float32, (9,), "Kinect intrinsics"
        ).reshape(3, 3),
        "object_pose_camera": _as_array(
            metadata["object_pose_camera"],
            np.float32,
            (16,),
            "Kinect object pose",
        ).reshape(4, 4),
        "pointcloud_xyz": points,
        "pointcloud_rgb": colors,
        "pointcloud_count": point_count,
        "frame_id": int(metadata["frame_id"]),
        "capture_monotonic_ns": int(metadata["capture_monotonic_ns"]),
        "capture_wall_ns": int(metadata["capture_wall_ns"]),
        "publish_monotonic_ns": int(metadata["publish_monotonic_ns"]),
        "inference_ms": float(metadata["inference_ms"]),
        "tracking_mode": str(metadata.get("tracking_mode", "foundationpose")),
        "sample_valid": bool(metadata.get("sample_valid", metadata["pose_valid"])),
        "pose_valid": bool(metadata["pose_valid"]),
        "segmentation_valid": bool(metadata.get("segmentation_valid", False)),
        "clouds_updated": bool(metadata["clouds_updated"]),
        "source_id": str(metadata["source_id"]),
    }


def pack_fixed_pointcloud(points, colors, capacity):
    """Return fixed-capacity point arrays plus the number of valid rows."""
    if capacity < 0:
        raise ValueError("point-cloud capacity cannot be negative")
    xyz = np.asarray(points, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.uint8)
    if xyz.ndim != 2 or xyz.shape[1:] != (3,) or rgb.shape != xyz.shape:
        raise ValueError("point-cloud arrays must have matching Nx3 shapes")
    if len(xyz) > capacity:
        selected = np.linspace(0, len(xyz) - 1, capacity, dtype=np.int64)
        xyz, rgb = xyz[selected], rgb[selected]
    count = len(xyz)
    fixed_xyz = np.zeros((capacity, 3), dtype=np.float32)
    fixed_rgb = np.zeros((capacity, 3), dtype=np.uint8)
    fixed_xyz[:count] = xyz
    fixed_rgb[:count] = rgb
    return fixed_xyz, fixed_rgb, count


class KinectZmqReceiver:
    """Background SUB receiver exposing the most recent complete Kinect sample."""

    def __init__(self, address, *, max_age_s=0.2, pointcloud_capacity=8192):
        if max_age_s <= 0:
            raise ValueError("Kinect max_age_s must be positive")
        if pointcloud_capacity < 0:
            raise ValueError("Kinect pointcloud_capacity cannot be negative")
        self.address = str(address)
        self.max_age_ns = int(float(max_age_s) * 1e9)
        self.pointcloud_capacity = int(pointcloud_capacity)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.setsockopt(zmq.SUBSCRIBE, KINECT_TOPIC)
        self._socket.connect(self.address)
        self._lock = threading.Lock()
        self._latest = None
        self._running = True
        self._error_count = 0
        self._thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._thread.start()

    def _receive_loop(self):
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            try:
                if not dict(poller.poll(100)).get(self._socket):
                    continue
                parts = self._socket.recv_multipart()
                # Drain queued samples so the control process always sees the newest one.
                while True:
                    try:
                        parts = self._socket.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                sample = decode_kinect_message(parts)
                sample["receive_monotonic_ns"] = time.monotonic_ns()
                xyz, rgb, count = pack_fixed_pointcloud(
                    sample["pointcloud_xyz"],
                    sample["pointcloud_rgb"],
                    self.pointcloud_capacity,
                )
                sample["pointcloud_xyz"] = xyz
                sample["pointcloud_rgb"] = rgb
                sample["pointcloud_count"] = count
                with self._lock:
                    self._latest = sample
            except (ValueError, KeyError, TypeError, zmq.ZMQError):
                self._error_count += 1

    def get_frames(self):
        with self._lock:
            sample = self._latest
        if sample is None:
            return {}
        now_ns = time.monotonic_ns()
        age_ns = now_ns - sample["capture_monotonic_ns"]
        result = dict(sample)
        result["sync_age_ms"] = age_ns / 1e6
        result["valid"] = bool(
            result.get("sample_valid", result["pose_valid"])
            and 0 <= age_ns <= self.max_age_ns
        )
        return {"KINECT": result}

    @property
    def error_count(self):
        return self._error_count

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._socket.close(0)
        self._context.term()
