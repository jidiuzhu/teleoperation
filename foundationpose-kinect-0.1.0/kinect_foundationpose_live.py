#!/usr/bin/env python3
"""Capture aligned Azure Kinect RGB-D and stream it to local FoundationPose C++."""
import argparse
import json
import socket
import struct
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from stream_client import azure_frames


FRAME_HEADER = struct.Struct("<4sIQIII9fQQQ")
POSE_RESPONSE = struct.Struct("<4sIIQf16f")
FLAG_INITIALIZE = 1
FLAG_STOP = 2
STATUS_MESSAGES = {
    1: "FoundationPose registration failed",
    2: "FoundationPose tracking failed",
    3: "FoundationPose rejected the frame",
    4: "FoundationPose hybrid re-registration rejected the mask",
}


class FoundationPoseResponseError(RuntimeError):
    """A valid FoundationPose response that reports a non-zero status."""

    def __init__(self, status):
        self.status = int(status)
        super().__init__(STATUS_MESSAGES.get(
            self.status, f"FoundationPose error status {self.status}"))


class RunningStats:
    """Constant-memory timing statistics for an unlimited live run."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.maximum = None

    def add(self, value):
        value = float(value)
        self.count += 1
        self.total += value
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    @property
    def mean(self):
        return self.total / self.count if self.count else None


def recv_exact(sock, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("FoundationPose server closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def replay_frames(recording):
    metadata_path = recording / "meta.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Replay metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    K = np.asarray(metadata["K"], dtype=np.float32)
    rgb_paths = sorted((recording / "rgb").glob("*.png"))
    if not rgb_paths:
        raise RuntimeError(f"Replay contains no RGB frames: {recording / 'rgb'}")
    for rgb_path in rgb_paths:
        depth_path = recording / "depth" / f"{rgb_path.stem}.npy"
        if not depth_path.is_file():
            raise RuntimeError(f"Replay depth is missing: {depth_path}")
        depth_m = np.load(depth_path)
        depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise RuntimeError(f"Replay RGB cannot be decoded: {rgb_path}")
        yield rgb, depth_mm, 0.001, K


def validate_geometry(rgb, depth, K, label):
    if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"{label} RGB must be an HxWx3 BGR image")
    if depth is None or depth.dtype != np.uint16 or depth.shape != rgb.shape[:2]:
        raise RuntimeError(
            f"{label} depth must be uint16 millimeters aligned to RGB; "
            f"got {None if depth is None else (depth.shape, depth.dtype)}")
    matrix = np.asarray(K, dtype=np.float32)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or matrix[2, 2] == 0:
        raise RuntimeError(f"{label} camera intrinsics are invalid")
    if np.count_nonzero(depth) == 0:
        raise RuntimeError(f"{label} aligned depth is empty")


def capture_initial(output, warmup, replay=None):
    output.mkdir(parents=True, exist_ok=True)
    frames = replay_frames(replay) if replay else azure_frames(fps=30, warmup=warmup)
    try:
        rgb, depth, _, K = next(frames)
    finally:
        frames.close()
    validate_geometry(rgb, depth, K, "Initialization")
    cv2.imwrite(str(output / "rgb.png"), rgb)
    cv2.imwrite(str(output / "depth.png"), depth)
    (output / "camera.json").write_text(
        json.dumps({"K": np.asarray(K).tolist()}, indent=2), encoding="utf-8")
    source = "replay" if replay else "Kinect"
    print(f"Captured {source} initialization frame: {rgb.shape[1]}x{rgb.shape[0]}")


def capture_sequence(output, warmup, replay, max_frames, frame_step, headless):
    """Capture a sparse RGB-D sequence while the object is rotated by hand."""
    rgb_dir, depth_dir = output / "rgb", output / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    frames = replay_frames(replay) if replay else azure_frames(fps=30, warmup=warmup)
    saved = 0
    source_frames = 0
    K_first = None
    print("Keep the camera fixed. Slowly rotate the object through two full views; "
          "change its tilt for the second view. Press Q when coverage is complete.", flush=True)
    try:
        for rgb, depth, _, K in frames:
            source_frames += 1
            validate_geometry(rgb, depth, K, f"Capture frame {source_frames}")
            if K_first is None:
                K_first = np.asarray(K, dtype=np.float32)
            elif not np.allclose(K_first, K, rtol=0, atol=1e-3):
                raise RuntimeError("Camera intrinsics changed during model-free capture")
            if (source_frames - 1) % frame_step == 0:
                name = f"{saved:06d}"
                if not cv2.imwrite(str(rgb_dir / f"{name}.png"), rgb):
                    raise RuntimeError("Failed to write model-free RGB frame")
                np.save(depth_dir / f"{name}.npy", depth.astype(np.float32) * 0.001)
                saved += 1
            if not headless:
                view = rgb.copy()
                cv2.putText(view, f"MODEL CAPTURE  saved {saved}/{max_frames or 'manual'}",
                            (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.imshow("Model-free object capture", view)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            if max_frames > 0 and saved >= max_frames:
                break
    finally:
        frames.close()
        cv2.destroyAllWindows()
    if saved < 3:
        raise RuntimeError(f"Model-free capture needs at least 3 frames; got {saved}")
    metadata = {
        "format": "foundationpose-model-free-v1",
        "source": "replay" if replay else "azure-kinect",
        "K": K_first.tolist(),
        "depth_unit": "meter",
        "frame_count": saved,
        "frame_step": frame_step,
        "capture_instruction": "fixed camera; object rotated by hand",
    }
    (output / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Captured model-free sequence: {saved} frames -> {output}")


def send_frame(sock, frame_id, rgb, depth, K, mask=None, flags=0):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    depth = np.ascontiguousarray(depth, dtype=np.uint16)
    if rgb.shape[:2] != depth.shape:
        raise ValueError("RGB and aligned depth dimensions differ")
    mask_bytes = b"" if mask is None else np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
    rgb_bytes = rgb.tobytes()
    depth_bytes = depth.tobytes()
    height, width = depth.shape
    header = FRAME_HEADER.pack(
        b"FPLV", 1, frame_id, width, height, flags,
        *np.asarray(K, dtype=np.float32).reshape(-1),
        len(rgb_bytes), len(depth_bytes), len(mask_bytes))
    sock.sendall(header)
    sock.sendall(rgb_bytes)
    sock.sendall(depth_bytes)
    if mask_bytes:
        sock.sendall(mask_bytes)

    values = POSE_RESPONSE.unpack(recv_exact(sock, POSE_RESPONSE.size))
    magic, version, status, response_id, processing_ms = values[:5]
    if magic != b"FPOS" or version != 1 or response_id != frame_id:
        raise RuntimeError(
            f"Invalid FoundationPose response for frame {frame_id}: "
            f"magic={magic!r} version={version} response_id={response_id}")
    if status != 0:
        raise FoundationPoseResponseError(status)
    return np.asarray(values[5:], np.float32).reshape(4, 4), processing_ms


def send_stop(sock, frame_id):
    header = FRAME_HEADER.pack(b"FPLV", 1, frame_id, 0, 0, FLAG_STOP,
                               *([0.0] * 9), 0, 0, 0)
    sock.sendall(header)


def load_obj_model(path, surface_points=12000):
    vertices = []
    faces = []
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(v) for v in line.split()[1:4]])
            elif line.startswith("f "):
                indices = []
                for token in line.split()[1:]:
                    index = int(token.split("/", 1)[0])
                    indices.append(index - 1 if index > 0 else len(vertices) + index)
                for i in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))
    points = np.asarray(vertices, np.float32)
    if points.size == 0:
        raise RuntimeError(f"Mesh has no vertices: {path}")
    bounds = (points.min(axis=0), points.max(axis=0))
    triangles = points[np.asarray(faces, np.int32)] if faces else np.empty((0, 3, 3), np.float32)
    if len(triangles) == 0:
        surface = points
    else:
        edges_a = triangles[:, 1] - triangles[:, 0]
        edges_b = triangles[:, 2] - triangles[:, 0]
        areas = np.linalg.norm(np.cross(edges_a, edges_b), axis=1) * 0.5
        valid = areas > 1e-12
        triangles, areas = triangles[valid], areas[valid]
        rng = np.random.default_rng(0)
        selected = rng.choice(len(triangles), surface_points, p=areas / areas.sum())
        u, v = rng.random((2, surface_points))
        reflected = u + v > 1
        u[reflected], v[reflected] = 1 - u[reflected], 1 - v[reflected]
        sampled = triangles[selected]
        surface = (sampled[:, 0] + u[:, None] * (sampled[:, 1] - sampled[:, 0]) +
                   v[:, None] * (sampled[:, 2] - sampled[:, 0])).astype(np.float32)
    return bounds, surface


def load_obj_bounds(path):
    return load_obj_model(path, surface_points=100)[0]


def project(K, pose, points):
    homogeneous = np.column_stack((points, np.ones(len(points), np.float32)))
    camera = (pose @ homogeneous.T).T[:, :3]
    uvw = (K @ camera.T).T
    return np.rint(uvw[:, :2] / uvw[:, 2:3]).astype(np.int32)


def draw_pose(image, K, pose, bounds):
    low, high = bounds
    corners = np.asarray([
        [low[0], low[1], low[2]], [high[0], low[1], low[2]],
        [high[0], high[1], low[2]], [low[0], high[1], low[2]],
        [low[0], low[1], high[2]], [high[0], low[1], high[2]],
        [high[0], high[1], high[2]], [low[0], high[1], high[2]],
    ], np.float32)
    pixels = project(K, pose, corners)
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
                 (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)):
        cv2.line(image, tuple(pixels[a]), tuple(pixels[b]), (0, 255, 0), 2)
    center = (low + high) * 0.5
    axis_len = float(np.mean(high - low) * 0.5)
    axes = np.vstack((center, center + [axis_len, 0, 0],
                      center + [0, axis_len, 0], center + [0, 0, axis_len]))
    axis_pixels = project(K, pose, axes.astype(np.float32))
    for endpoint, color in zip(axis_pixels[1:], ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(image, tuple(axis_pixels[0]), tuple(endpoint), color, 3)
    return image


def depth_to_pointcloud(depth_mm, K, bgr, stride=3, min_depth=0.15,
                        max_depth=2.0, max_points=120000, pixel_mask=None):
    """Back-project aligned millimeter depth into a filtered colored cloud."""
    depth = depth_mm[::stride, ::stride].astype(np.float32) * 0.001
    colors = bgr[::stride, ::stride, ::-1]
    rows, cols = np.indices(depth.shape, dtype=np.float32)
    u = cols * stride
    v = rows * stride
    valid = np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
    if pixel_mask is not None:
        valid &= pixel_mask[::stride, ::stride].astype(bool)
    z = depth[valid]
    x = (u[valid] - float(K[0, 2])) * z / float(K[0, 0])
    y = (v[valid] - float(K[1, 2])) * z / float(K[1, 1])
    points = np.column_stack((x, y, z)).astype(np.float32)
    rgb = colors[valid].astype(np.uint8)
    if len(points) > max_points:
        indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
        points, rgb = points[indices], rgb[indices]
    return points, rgb


def observed_object_cloud(depth_mm, K, bgr, pose, bounds, args, pixel_mask=None):
    """Crop aligned depth to the posed object's expanded model-space OBB."""
    points, colors = depth_to_pointcloud(
        depth_mm, K, bgr, args.pointcloud_stride, args.pointcloud_min_depth,
        args.pointcloud_max_depth, args.pointcloud_max_points, pixel_mask=pixel_mask)
    if len(points) == 0:
        return points, colors
    low, high = bounds
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    margin = args.pointcloud_crop_margin
    keep = ((local >= low - margin) & (local <= high + margin)).all(axis=1)
    return points[keep], colors[keep]


def transform_model_cloud(points, pose):
    return points @ pose[:3, :3].T + pose[:3, 3]


class ReferenceCloud:
    """Fuse observed target points into a fixed cloud in the model frame."""

    def __init__(self, frames, voxel):
        self.frames = frames
        self.voxel = voxel
        self.samples = []
        self.colors = []
        self.model_points = None
        self.model_colors = None

    @property
    def ready(self):
        return self.model_points is not None

    def add(self, points_camera, colors, pose):
        if self.ready or len(points_camera) == 0:
            return
        self.samples.append((points_camera - pose[:3, 3]) @ pose[:3, :3])
        self.colors.append(colors)
        if len(self.samples) < self.frames:
            return
        points = np.concatenate(self.samples)
        colors = np.concatenate(self.colors).astype(np.float32)
        voxels = np.floor(points / self.voxel).astype(np.int64)
        _, inverse = np.unique(voxels, axis=0, return_inverse=True)
        counts = np.bincount(inverse)
        merged_points = np.zeros((len(counts), 3), np.float64)
        merged_colors = np.zeros((len(counts), 3), np.float64)
        np.add.at(merged_points, inverse, points)
        np.add.at(merged_colors, inverse, colors)
        self.model_points = (merged_points / counts[:, None]).astype(np.float32)
        self.model_colors = np.uint8(np.clip(merged_colors / counts[:, None], 0, 255))
        self.samples = self.colors = None

    def at_pose(self, pose):
        if self.ready:
            return transform_model_cloud(self.model_points, pose), self.model_colors
        if not self.samples:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
        points = np.concatenate(self.samples)
        colors = np.concatenate(self.colors)
        return transform_model_cloud(points, pose), colors


def select_clouds(mode, observed, mesh, reference):
    selected = []
    if mode in ("all", "observed") and len(observed[0]):
        selected.append((observed[0], np.tile(np.array([[55, 135, 255]], np.uint8),
                                                    (len(observed[0]), 1))))
    if mode in ("all", "mesh") and len(mesh[0]):
        selected.append((mesh[0], np.tile(np.array([[255, 65, 50]], np.uint8),
                                                (len(mesh[0]), 1))))
    if mode in ("all", "reference") and len(reference[0]):
        selected.append((reference[0], np.tile(np.array([[255, 215, 35]], np.uint8),
                                                     (len(reference[0]), 1))))
    if not selected:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint8)
    return np.concatenate([item[0] for item in selected]), np.concatenate(
        [item[1] for item in selected])


def render_pointcloud(points, colors, size, yaw_deg=-25.0, pitch_deg=-20.0):
    """Render a lightweight orthographic cloud view without an OpenGL dependency."""
    width, height = size
    canvas = np.full((height, width, 3), 24, np.uint8)
    if len(points) == 0:
        return canvas
    yaw, pitch = np.deg2rad([yaw_deg, pitch_deg])
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    rotation = np.array([[cy, 0, sy], [sp * sy, cp, -sp * cy],
                         [-cp * sy, sp, cp * cy]], np.float32)
    center = np.median(points, axis=0)
    rotated = (points - center) @ rotation.T
    lo = np.percentile(rotated[:, :2], 1, axis=0)
    hi = np.percentile(rotated[:, :2], 99, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = 0.9 * min(width / span[0], height / span[1])
    pixels = np.empty((len(points), 2), np.int32)
    pixels[:, 0] = np.rint((rotated[:, 0] - (lo[0] + hi[0]) * 0.5) * scale
                           + width * 0.5).astype(np.int32)
    pixels[:, 1] = np.rint((rotated[:, 1] - (lo[1] + hi[1]) * 0.5) * scale
                           + height * 0.5).astype(np.int32)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
              (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
    pixels, bgr = pixels[inside], colors[inside, ::-1]
    order = np.argsort(rotated[inside, 2])[::-1]
    pixels, bgr = pixels[order], bgr[order]
    canvas[pixels[:, 1], pixels[:, 0]] = bgr
    return canvas


def depth_preview(depth_mm, min_depth, max_depth):
    depth = depth_mm.astype(np.float32) * 0.001
    normalized = np.clip((depth - min_depth) / (max_depth - min_depth), 0, 1)
    image = cv2.applyColorMap(np.uint8((1.0 - normalized) * 255), cv2.COLORMAP_TURBO)
    image[depth_mm == 0] = 0
    return image


def compose_preview(rgb_pose, depth, cloud, args):
    if args.visualization_layout == "rgb":
        return rgb_pose
    scale = args.preview_scale
    panel_size = (max(1, int(rgb_pose.shape[1] * scale)),
                  max(1, int(rgb_pose.shape[0] * scale)))
    rgb_panel = cv2.resize(rgb_pose, panel_size, interpolation=cv2.INTER_AREA)
    depth_panel = cv2.resize(
        depth_preview(depth, args.pointcloud_min_depth, args.pointcloud_max_depth),
        panel_size, interpolation=cv2.INTER_AREA)
    panels = [("RGB + POSE", rgb_panel), ("ALIGNED DEPTH", depth_panel)]
    if args.visualization_layout == "rgbd_cloud":
        cloud_panel = (np.full_like(rgb_panel, 24) if cloud is None else
                       render_pointcloud(cloud[0], cloud[1], panel_size,
                                         args.pointcloud_yaw, args.pointcloud_pitch))
        panels.append((f"TARGET CLOUD: {args.pointcloud_mode.upper()}", cloud_panel))
    for label, panel in panels:
        cv2.putText(panel, label, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return np.hstack([panel for _label, panel in panels])


def load_initialization(init_dir):
    rgb = cv2.imread(str(init_dir / "rgb.png"))
    depth = cv2.imread(str(init_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(init_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    camera_path = init_dir / "camera.json"
    if not camera_path.is_file():
        raise RuntimeError(f"Initialization camera metadata is missing: {camera_path}")
    K = np.asarray(json.loads(camera_path.read_text(encoding="utf-8"))["K"],
                   dtype=np.float32)
    validate_geometry(rgb, depth, K, "Initialization")
    if mask is None or mask.shape != depth.shape:
        raise RuntimeError("Initialization mask is missing or does not match RGB-D dimensions")
    foreground = int(np.count_nonzero(mask))
    if foreground == 0:
        raise RuntimeError("Initialization mask is empty; FoundationPose was not started")
    print(f"Initialization valid: {rgb.shape[1]}x{rgb.shape[0]}, mask={foreground} pixels")
    return rgb, depth, (mask > 0).astype(np.uint8) * 255, K


def mask_pose_center_error_px(mask, K, pose, bounds):
    """Compare the SAM mask center with the projected center of the posed mesh."""
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return float("inf")
    mask_center = np.array([np.median(cols), np.median(rows)], np.float32)
    model_center = np.append((bounds[0] + bounds[1]) * 0.5, 1.0)
    camera_center = np.asarray(pose, dtype=np.float32) @ model_center
    if not np.isfinite(camera_center).all() or camera_center[2] <= 0:
        return float("inf")
    projected_h = np.asarray(K, dtype=np.float32) @ camera_center[:3]
    projected = projected_h[:2] / projected_h[2]
    if not np.isfinite(projected).all():
        return float("inf")
    return float(np.linalg.norm(mask_center - projected))


class HybridSamController:
    """Run bounded asynchronous SAM updates and request recovery registrations."""

    def __init__(self, args, init_mask, bounds):
        from server.sam3_live_worker import LiveSamWorker

        prompts = [value.strip() for value in args.sam_prompts.split(",") if value.strip()]
        if not prompts:
            raise ValueError("hybrid SAM requires at least one non-empty prompt")
        self.args = args
        self.bounds = bounds
        self.worker = LiveSamWorker(
            args.sam_checkpoint, args.sam_bpe, prompts, args.sam_threshold,
            args.hybrid_sam_hz, args.hybrid_window_seconds,
            args.hybrid_min_mask_pixels, args.hybrid_max_area_change,
            seed_mask=init_mask)
        self.frames = OrderedDict()
        self.max_cached_frames = max(
            2, int(np.ceil(args.hybrid_sam_hz * args.hybrid_window_seconds)) + 1)
        self.next_submit_at = 0.0
        self.last_reregister_at = -float("inf")
        self.started_at = time.monotonic()
        self.status = "SAM3 loading"
        self.failed = False
        self.submitted = 0
        self.replaced = 0
        self.results = 0
        self.errors = 0
        self.stale = 0
        self.reregistrations = 0
        self.sam_times = RunningStats()
        self.result_ages = RunningStats()
        self.reregister_times = RunningStats()
        print(
            f"HYBRID enabled: SAM3 target={args.hybrid_sam_hz:.2f} Hz "
            f"window={args.hybrid_window_seconds:.2f}s; FoundationPose remains per-frame",
            flush=True)

    def _prune(self, now):
        cutoff = now - self.args.hybrid_window_seconds
        while self.frames:
            _frame_id, sample = next(iter(self.frames.items()))
            if sample["captured_at"] >= cutoff:
                break
            self.frames.popitem(last=False)
        while len(self.frames) > self.max_cached_frames:
            self.frames.popitem(last=False)

    def submit_if_due(self, frame_id, captured_at, rgb, depth, K, pose):
        if self.failed or captured_at < self.next_submit_at:
            return
        interval = 1.0 / self.args.hybrid_sam_hz
        self.next_submit_at = max(self.next_submit_at + interval, captured_at + interval)
        sample_bgr = rgb.copy()
        sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
        self.frames[frame_id] = {
            "captured_at": captured_at,
            "rgb": sample_bgr,
            "depth": depth.copy(),
            "K": np.asarray(K, dtype=np.float32).copy(),
            "pose": np.asarray(pose, dtype=np.float32).copy(),
        }
        self._prune(captured_at)
        if self.worker.submit(frame_id, captured_at, sample_rgb):
            self.replaced += 1
        self.submitted += 1

    def consume(self, sock, pose):
        now = time.monotonic()
        self._prune(now)
        for message in self.worker.poll():
            kind = message.get("kind")
            if kind == "ready":
                self.status = f"SAM3 ready {message['load_ms'] / 1000.0:.1f}s"
                print(
                    f"HYBRID SAM3 ready: load={message['load_ms']:.0f} ms "
                    f"GPU={message['gpu_allocated_mib']}/{message['gpu_reserved_mib']} MiB "
                    "allocated/reserved", flush=True)
                continue
            if kind == "fatal":
                self.failed = True
                self.frames.clear()
                self.errors += 1
                self.status = "SAM3 disabled after worker failure"
                print("HYBRID SAM3 fatal error; continuing FoundationPose-only:\n" +
                      message.get("traceback", message.get("message", "unknown error")),
                      flush=True)
                continue
            if kind == "frame_error":
                self.errors += 1
                self.status = "SAM3 frame rejected"
                print(f"HYBRID SAM3 frame {message.get('frame_id')} rejected: "
                      f"{message.get('message')}", flush=True)
                continue
            if kind != "mask":
                continue

            self.results += 1
            self.sam_times.add(message["inference_ms"])
            frame_id = int(message["frame_id"])
            sample = self.frames.pop(frame_id, None)
            age = now - float(message["captured_at"])
            self.result_ages.add(age * 1000.0)
            if sample is None or age > self.args.hybrid_window_seconds:
                self.stale += 1
                self.status = f"SAM3 stale {age * 1000.0:.0f} ms"
                continue

            mask = np.asarray(message["mask"], dtype=np.uint8)
            if mask.shape != sample["depth"].shape:
                self.errors += 1
                self.status = "SAM3 invalid mask shape"
                continue
            center_error = mask_pose_center_error_px(
                mask, sample["K"], sample["pose"], self.bounds)
            action = "monitor"
            if (center_error >= self.args.hybrid_reregister_center_px and
                    now - self.last_reregister_at >=
                    self.args.hybrid_reregister_cooldown):
                self.last_reregister_at = now
                register_started = time.perf_counter()
                try:
                    pose, register_ms = send_frame(
                        sock, frame_id, sample["rgb"], sample["depth"], sample["K"],
                        mask * 255, FLAG_INITIALIZE)
                except FoundationPoseResponseError as exc:
                    if exc.status != 4:
                        raise
                    self.errors += 1
                    action = "reregister-rejected"
                    print(f"HYBRID re-registration rejected at frame {frame_id}: {exc}",
                          flush=True)
                else:
                    wall_ms = (time.perf_counter() - register_started) * 1000.0
                    self.reregister_times.add(register_ms)
                    self.reregistrations += 1
                    action = f"reregister {register_ms:.0f} ms/{wall_ms:.0f} ms wall"

            self.status = (
                f"SAM3 {message['inference_ms']:.0f} ms age {age * 1000.0:.0f} ms "
                f"offset {center_error:.0f}px {action}")
            if self.results % 10 == 0 or action != "monitor":
                print(
                    f"HYBRID frame={frame_id} sam={message['inference_ms']:.1f} ms "
                    f"age={age * 1000.0:.0f} ms center_error={center_error:.1f}px "
                    f"action={action}", flush=True)
        if not self.failed and not self.worker.alive:
            self.failed = True
            self.frames.clear()
            self.errors += 1
            self.status = "SAM3 worker stopped"
            print("HYBRID SAM3 worker stopped; continuing FoundationPose-only", flush=True)
        return pose

    def summary(self, elapsed):
        return {
            "enabled": True,
            "target_sam_hz": self.args.hybrid_sam_hz,
            "window_seconds": self.args.hybrid_window_seconds,
            "submitted_frames": self.submitted,
            "replaced_queued_frames": self.replaced,
            "mask_results": self.results,
            "effective_sam_hz": self.results / elapsed if elapsed > 0 else None,
            "sam_mean_ms": self.sam_times.mean,
            "sam_max_ms": self.sam_times.maximum,
            "result_age_mean_ms": self.result_ages.mean,
            "cached_frame_limit": self.max_cached_frames,
            "cached_frames_at_exit": len(self.frames),
            "stale_results": self.stale,
            "errors": self.errors,
            "reregistrations": self.reregistrations,
            "reregister_mean_ms": self.reregister_times.mean,
            "worker_failed": self.failed,
        }

    def close(self):
        self.worker.close()


def stream(args):
    init_dir = Path(args.init_dir)
    init_rgb, init_depth, init_mask, K = load_initialization(init_dir)
    bounds, mesh_model = load_obj_model(Path(args.mesh), args.pointcloud_mesh_points)

    pointcloud_dir = Path(args.pointcloud_output) if args.pointcloud_output else None
    try:
        connection = socket.create_connection(
            (args.host, args.port), timeout=args.connect_timeout)
    except OSError as exc:
        raise ConnectionError(
            f"Cannot connect to FoundationPose at {args.host}:{args.port}; "
            "inspect the retained container log") from exc

    with connection as sock:
        sock.settimeout(args.register_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            pose, inference_ms = send_frame(sock, 0, init_rgb, init_depth, K, init_mask,
                                            FLAG_INITIALIZE)
        except socket.timeout as exc:
            raise TimeoutError(
                f"FoundationPose Register exceeded {args.register_timeout:.0f}s") from exc
        register_ms = float(inference_ms)
        print(f"REGISTER inference={inference_ms:.1f} ms")
        sock.settimeout(args.track_timeout)

        reference_cloud = ReferenceCloud(args.pointcloud_reference_frames,
                                         args.pointcloud_reference_voxel)
        initial_observed = observed_object_cloud(
            init_depth, K, init_rgb, pose, bounds, args, pixel_mask=init_mask)
        reference_cloud.add(*initial_observed, pose)

        frames = (replay_frames(Path(args.replay)) if args.replay
                  else azure_frames(fps=30, warmup=args.warmup))
        started = time.perf_counter()
        count = 0
        track_times = RunningStats()
        cached_cloud = None
        cloud_counts = {"observed": 0, "mesh": len(mesh_model), "reference": 0}
        hybrid = None
        try:
            hybrid = (HybridSamController(args, init_mask, bounds)
                      if args.hybrid_enabled else None)
            for rgb, depth, _, live_K in frames:
                count += 1
                captured_at = time.monotonic()
                validate_geometry(rgb, depth, live_K, f"Frame {count}")
                if hybrid is not None:
                    pose = hybrid.consume(sock, pose)
                try:
                    pose, inference_ms = send_frame(sock, count, rgb, depth, live_K)
                except socket.timeout as exc:
                    raise TimeoutError(
                        f"FoundationPose Track frame {count} exceeded "
                        f"{args.track_timeout:.0f}s") from exc
                track_times.add(inference_ms)
                if hybrid is not None:
                    hybrid.submit_if_due(
                        count, captured_at, rgb, depth, live_K, pose)
                needs_cloud = (args.pointcloud_enabled and
                               (count == 1 or count % args.pointcloud_preview_every == 0 or
                                (args.pointcloud_save_every > 0 and
                                 count % args.pointcloud_save_every == 0)))
                if needs_cloud:
                    observed = observed_object_cloud(depth, live_K, rgb, pose, bounds, args)
                    reference_cloud.add(*observed, pose)
                    mesh_cloud = (transform_model_cloud(mesh_model, pose),
                                  np.empty((len(mesh_model), 3), np.uint8))
                    reference = reference_cloud.at_pose(pose)
                    cached_cloud = select_clouds(args.pointcloud_mode, observed, mesh_cloud,
                                                 reference)
                    cloud_counts = {"observed": len(observed[0]), "mesh": len(mesh_model),
                                    "reference": len(reference[0])}
                if args.pose_output:
                    pose_path = Path(args.pose_output)
                    pose_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = pose_path.with_suffix(pose_path.suffix + ".tmp")
                    np.savetxt(temporary, pose, fmt="%.9g")
                    temporary.replace(pose_path)
                if not args.headless:
                    view = draw_pose(rgb.copy(), live_K, pose, bounds)
                    cv2.putText(view, f"FP {inference_ms:.1f} ms", (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if hybrid is not None:
                        cv2.putText(view, hybrid.status, (20, 68),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                                    (0, 255, 255), 2, cv2.LINE_AA)
                    view = compose_preview(view, depth, cached_cloud, args)
                    cv2.imshow("Kinect FoundationPose", view)
                    key = cv2.waitKey(1) & 0xFF
                    mode_keys = {ord("0"): "all", ord("1"): "observed",
                                 ord("2"): "mesh", ord("3"): "reference"}
                    if key in mode_keys:
                        args.pointcloud_mode = mode_keys[key]
                        cached_cloud = None
                    if key in (27, ord("q")):
                        break
                if count % 30 == 0:
                    fps = count / (time.perf_counter() - started)
                    print(f"TRACK frame={count} inference={inference_ms:.2f} ms stream={fps:.1f} fps")
                if args.max_frames > 0 and count >= args.max_frames:
                    break
        finally:
            frames.close()
            cv2.destroyAllWindows()
            if hybrid is not None:
                hybrid.close()
            try:
                send_stop(sock, count + 1)
            except OSError:
                pass
        elapsed = time.perf_counter() - started
        summary = {
            "register_ms": register_ms,
            "track_frames": count,
            "track_mean_ms": track_times.mean,
            "track_max_ms": track_times.maximum,
            "stream_fps": count / elapsed if elapsed > 0 else None,
            "final_pose_is_identity": bool(np.allclose(pose, np.eye(4), atol=1e-5)),
            "pointcloud_files": (len(list(pointcloud_dir.rglob("*.ply")))
                                 if pointcloud_dir and pointcloud_dir.is_dir() else 0),
            "last_pointcloud_points": (len(cached_cloud[0]) if cached_cloud is not None else 0),
            "target_cloud_points": cloud_counts,
            "reference_cloud_ready": reference_cloud.ready,
            "hybrid": (hybrid.summary(elapsed) if hybrid is not None else
                       {"enabled": False}),
        }
        if args.metrics_output:
            metrics_path = Path(args.metrics_output)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("SUMMARY " + json.dumps(summary))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--output", required=True)
    capture.add_argument("--warmup", type=int, default=10)
    capture.add_argument("--replay")
    capture_sequence_parser = subparsers.add_parser("capture-sequence")
    capture_sequence_parser.add_argument("--output", required=True)
    capture_sequence_parser.add_argument("--warmup", type=int, default=10)
    capture_sequence_parser.add_argument("--replay")
    capture_sequence_parser.add_argument("--max-frames", type=int, default=240)
    capture_sequence_parser.add_argument("--frame-step", type=int, default=3)
    capture_sequence_parser.add_argument("--headless", action="store_true")
    live = subparsers.add_parser("stream")
    live.add_argument("--init-dir", required=True)
    live.add_argument("--mesh", required=True)
    live.add_argument("--host", default="127.0.0.1")
    live.add_argument("--port", type=int, default=50051)
    live.add_argument("--warmup", type=int, default=10)
    live.add_argument("--max-frames", type=int, default=0)
    live.add_argument("--headless", action="store_true")
    live.add_argument("--replay")
    live.add_argument("--connect-timeout", type=float, default=60)
    live.add_argument("--register-timeout", type=float, default=300)
    live.add_argument("--track-timeout", type=float, default=60)
    live.add_argument("--pose-output")
    live.add_argument("--metrics-output")
    live.add_argument("--visualization-layout", choices=("rgb", "rgbd", "rgbd_cloud"),
                      default="rgb")
    live.add_argument("--preview-scale", type=float, default=0.5)
    live.add_argument("--pointcloud-enabled", action="store_true")
    live.add_argument("--pointcloud-mode", choices=("all", "observed", "mesh", "reference"),
                      default="all")
    live.add_argument("--pointcloud-output")
    live.add_argument("--pointcloud-save-every", type=int, default=0)
    live.add_argument("--pointcloud-preview-every", type=int, default=2)
    live.add_argument("--pointcloud-stride", type=int, default=3)
    live.add_argument("--pointcloud-min-depth", type=float, default=0.15)
    live.add_argument("--pointcloud-max-depth", type=float, default=2.0)
    live.add_argument("--pointcloud-max-points", type=int, default=120000)
    live.add_argument("--pointcloud-mesh-points", type=int, default=12000)
    live.add_argument("--pointcloud-reference-frames", type=int, default=5)
    live.add_argument("--pointcloud-reference-voxel", type=float, default=0.003)
    live.add_argument("--pointcloud-crop-margin", type=float, default=0.005)
    live.add_argument("--pointcloud-yaw", type=float, default=-25.0)
    live.add_argument("--pointcloud-pitch", type=float, default=-20.0)
    live.add_argument("--hybrid-enabled", action="store_true")
    live.add_argument("--hybrid-sam-hz", type=float, default=5.0)
    live.add_argument("--hybrid-window-seconds", type=float, default=2.0)
    live.add_argument("--hybrid-reregister-center-px", type=float, default=40.0)
    live.add_argument("--hybrid-reregister-cooldown", type=float, default=2.0)
    live.add_argument("--hybrid-min-mask-pixels", type=int, default=500)
    live.add_argument("--hybrid-max-area-change", type=float, default=2.5)
    live.add_argument("--sam-checkpoint")
    live.add_argument("--sam-bpe")
    live.add_argument("--sam-prompts")
    live.add_argument("--sam-threshold", type=float, default=0.5)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--init-dir", required=True)
    args = parser.parse_args()
    if args.command == "capture":
        capture_initial(Path(args.output), args.warmup,
                        Path(args.replay) if args.replay else None)
    elif args.command == "capture-sequence":
        if args.max_frames < 0 or args.frame_step < 1:
            parser.error("capture-sequence requires max-frames >= 0 and frame-step >= 1")
        capture_sequence(Path(args.output), args.warmup,
                         Path(args.replay) if args.replay else None,
                         args.max_frames, args.frame_step, args.headless)
    elif args.command == "stream":
        if args.hybrid_enabled:
            if not args.sam_checkpoint or not args.sam_bpe or not args.sam_prompts:
                parser.error("hybrid mode requires SAM checkpoint, BPE, and prompts")
            if args.hybrid_sam_hz <= 0 or args.hybrid_window_seconds <= 0:
                parser.error("hybrid SAM rate and window must be positive")
            if (args.hybrid_reregister_center_px <= 0 or
                    args.hybrid_reregister_cooldown < 0 or
                    args.hybrid_min_mask_pixels < 1 or
                    args.hybrid_max_area_change <= 1):
                parser.error("invalid hybrid validation/re-registration settings")
        stream(args)
    else:
        load_initialization(Path(args.init_dir))


if __name__ == "__main__":
    main()
