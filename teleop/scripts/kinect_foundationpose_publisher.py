#!/usr/bin/env python3
"""Publish Azure Kinect RGB-D and FoundationPose results for teleop recording."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import time

import cv2
import numpy as np
import zmq


TELEOP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TELEOP_ROOT.parent
DEFAULT_FOUNDATION_ROOT = WORKSPACE_ROOT / "foundationpose-kinect-0.1.0"
DEFAULT_CAMERA_CALIBRATION = TELEOP_ROOT / "kinect_hand_eye.json"
DEFAULT_CAPTURE_DIR = WORKSPACE_ROOT / "data_captures"
DEFAULT_VIDEO_DIR = WORKSPACE_ROOT / "videos"
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

from utils.kinect_stream import encode_kinect_message


@dataclass(frozen=True)
class _TrackingResult:
    frame_id: int
    pose: np.ndarray
    inference_ms: float
    observed: object
    clouds_updated: bool
    segmentation_valid: bool


@dataclass(frozen=True)
class _CapturedFrame:
    frame_id: int
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    depth_scale: float
    intrinsics: np.ndarray
    monotonic_ns: int
    wall_ns: int


class LatestFrameSource:
    """Capture continuously and expose only the newest unprocessed frame."""

    def __init__(self, frames):
        self.frames = frames
        self.queue = queue.Queue(maxsize=1)
        self.stopping = threading.Event()
        self.finished = threading.Event()
        self.error = None
        self.captured = 0
        self.dropped = 0
        self.thread = threading.Thread(
            target=self._capture,
            name="kinect-latest-frame",
            daemon=True,
        )
        self.thread.start()

    def _capture(self):
        try:
            for color_bgr, depth_mm, depth_scale, intrinsics in self.frames:
                if self.stopping.is_set():
                    break
                self.captured += 1
                item = _CapturedFrame(
                    frame_id=self.captured,
                    color_bgr=color_bgr,
                    depth_mm=depth_mm,
                    depth_scale=depth_scale,
                    intrinsics=intrinsics,
                    monotonic_ns=time.monotonic_ns(),
                    wall_ns=time.time_ns(),
                )
                try:
                    self.queue.put_nowait(item)
                except queue.Full:
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                    else:
                        self.dropped += 1
                    self.queue.put_nowait(item)
        except BaseException as exc:  # surfaced in the publisher thread
            self.error = exc
        finally:
            try:
                self.frames.close()
            finally:
                self.finished.set()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                return self.queue.get(timeout=0.1)
            except queue.Empty:
                if not self.finished.is_set():
                    continue
                if self.error is not None:
                    raise RuntimeError("Kinect capture thread failed") from self.error
                raise StopIteration

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=5.0)


def render_segmentation_view(
    color_bgr,
    mask,
    *,
    valid,
    box=None,
    panel_size=None,
    show_mask=True,
):
    """Render RGB alone or beside the final mask at the requested size."""
    color = np.asarray(color_bgr)
    binary = np.asarray(mask) > 0
    if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("Viewer color frame must be HxWx3 uint8 BGR")
    if binary.shape != color.shape[:2]:
        raise ValueError("Viewer mask must match the color frame")
    if panel_size is None:
        width, height = color.shape[1], color.shape[0]
    else:
        width, height = (int(panel_size[0]), int(panel_size[1]))
    if width < 1 or height < 1:
        raise ValueError("Viewer panel dimensions must be positive")

    image_panel = cv2.resize(color, (width, height), interpolation=cv2.INTER_AREA)
    if box is not None:
        xyxy = np.asarray(box, dtype=np.float32).reshape(-1)
        if xyxy.shape != (4,) or not np.all(np.isfinite(xyxy)):
            raise ValueError("Viewer box must contain four finite XYXY values")
        scale_x = width / color.shape[1]
        scale_y = height / color.shape[0]
        x0, y0, x1, y1 = xyxy
        top_left = (
            int(np.clip(round(x0 * scale_x), 0, width - 1)),
            int(np.clip(round(y0 * scale_y), 0, height - 1)),
        )
        bottom_right = (
            int(np.clip(round(x1 * scale_x), 0, width - 1)),
            int(np.clip(round(y1 * scale_y), 0, height - 1)),
        )
        cv2.rectangle(image_panel, top_left, bottom_right, (0, 255, 255), 2)
    if not show_mask:
        return np.ascontiguousarray(image_panel)

    resized_mask = cv2.resize(
        binary.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    mask_panel = (image_panel.astype(np.float32) * 0.2).astype(np.uint8)
    if valid:
        selected = resized_mask > 0
        mask_panel[selected] = image_panel[selected]
        contours, _ = cv2.findContours(
            resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(mask_panel, contours, -1, (0, 255, 0), 2)
    return np.ascontiguousarray(np.hstack((image_panel, mask_panel)))


class RgbVideoRecorder:
    """Write full-resolution Kinect color frames to a timestamped MP4."""

    def __init__(self, output_dir, fps, *, writer_factory=None):
        self.output_dir = Path(output_dir).expanduser()
        self.fps = float(fps)
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("RGB video FPS must be positive")
        self.writer_factory = writer_factory or cv2.VideoWriter
        self.writer = None
        self.path = None
        self.frame_size = None
        self.frames_written = 0
        self.started_at = None

    @property
    def active(self):
        return self.writer is not None

    @staticmethod
    def _validated_frame(color_bgr):
        frame = np.asarray(color_bgr)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("RGB video frame must be HxWx3 uint8 BGR")
        return np.ascontiguousarray(frame)

    def start(self, color_bgr, *, captured_at=None):
        if self.active:
            raise RuntimeError("RGB video recording is already active")
        frame = self._validated_frame(color_bgr)
        height, width = frame.shape[:2]
        timestamp = captured_at or datetime.now().astimezone()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = timestamp.strftime("kinect_rgb_%Y%m%d_%H%M%S_%f.mp4")
        path = self.output_dir / name
        writer = self.writer_factory(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"OpenCV could not open RGB video: {path}")
        self.writer = writer
        self.path = path
        self.frame_size = (width, height)
        self.frames_written = 0
        self.started_at = time.monotonic()
        return path

    def write(self, color_bgr):
        if not self.active:
            return
        frame = self._validated_frame(color_bgr)
        height, width = frame.shape[:2]
        if (width, height) != self.frame_size:
            raise ValueError(
                "RGB video frame size changed from "
                f"{self.frame_size} to {(width, height)}"
            )
        self.writer.write(frame)
        self.frames_written += 1

    def stop(self):
        if not self.active:
            return None
        writer = self.writer
        summary = {
            "path": self.path,
            "frames": self.frames_written,
            "elapsed_s": time.monotonic() - self.started_at,
            "fps": self.fps,
            "frame_size": self.frame_size,
        }
        self.writer = None
        self.path = None
        self.frame_size = None
        self.frames_written = 0
        self.started_at = None
        writer.release()
        return summary


class SegmentationViewer:
    """Display the current Kinect image and Everything segmentation mask."""

    WINDOW_NAME = "Kinect Everything Viewer"
    MASK_TRACKBAR = "Mask"
    DEFAULT_WIDTH = 1280

    @staticmethod
    def is_save_key(key):
        return (int(key) & 0xFF) in (10, 13)

    def __init__(self, *, video_dir=DEFAULT_VIDEO_DIR, video_fps=30):
        if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        ):
            raise RuntimeError(
                "--viewer requires a graphical session; DISPLAY and "
                "WAYLAND_DISPLAY are both unset"
            )
        self.closed = False
        self.save_requested = False
        self.mask_visible = True
        self.video_recorder = RgbVideoRecorder(video_dir, video_fps)
        try:
            cv2.namedWindow(
                self.WINDOW_NAME,
                cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
            )
            cv2.resizeWindow(self.WINDOW_NAME, self.DEFAULT_WIDTH, 360)
            cv2.createTrackbar(
                self.MASK_TRACKBAR,
                self.WINDOW_NAME,
                1,
                1,
                self._on_mask_trackbar,
            )
        except cv2.error as exc:
            raise RuntimeError("OpenCV could not create the --viewer window") from exc

    def _window_width(self):
        try:
            _, _, width, _ = cv2.getWindowImageRect(self.WINDOW_NAME)
        except (cv2.error, AttributeError):
            width = self.DEFAULT_WIDTH
        return max(1, int(width))

    def _on_mask_trackbar(self, value):
        visible = bool(value)
        if visible == self.mask_visible:
            return
        width = self._window_width()
        self.mask_visible = visible
        aspect_width = 32 if visible else 16
        height = max(1, round(width * 9 / aspect_width))
        if not self.closed:
            cv2.resizeWindow(self.WINDOW_NAME, width, height)

    def _toggle_mask(self):
        value = 0 if self.mask_visible else 1
        cv2.setTrackbarPos(self.MASK_TRACKBAR, self.WINDOW_NAME, value)
        # Some OpenCV backends do not invoke callbacks synchronously.
        self._on_mask_trackbar(value)

    def _toggle_recording(self, color_bgr):
        if self.video_recorder.active:
            summary = self.video_recorder.stop()
            print(
                "[KinectViewer] RGB video saved: "
                f"{summary['path']} ({summary['frames']} frames, "
                f"{summary['elapsed_s']:.1f}s)",
                flush=True,
            )
            return
        path = self.video_recorder.start(color_bgr)
        self.video_recorder.write(color_bgr)
        print(
            f"[KinectViewer] RGB video recording started: {path}",
            flush=True,
        )

    @staticmethod
    def _label(view, text, origin, color):
        cv2.putText(
            view,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            view,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            1,
            cv2.LINE_AA,
        )

    def show(self, color_bgr, mask, *, valid, box=None):
        self.video_recorder.write(color_bgr)
        view = render_segmentation_view(
            color_bgr,
            mask,
            valid=valid,
            box=box,
            panel_size=(640, 360) if self.mask_visible else None,
            show_mask=self.mask_visible,
        )
        self._label(
            view,
            "CUTIE BOX" if box is not None else "RGB",
            (12, 28),
            (0, 255, 255) if box is not None else (255, 255, 255),
        )
        if self.mask_visible:
            self._label(
                view,
                "MASK" if valid else "MASK INVALID",
                (view.shape[1] // 2 + 12, 28),
                (0, 255, 0) if valid else (0, 0, 255),
            )
        if self.video_recorder.active:
            cv2.circle(view, (view.shape[1] - 74, 21), 7, (0, 0, 255), -1)
            self._label(
                view,
                "REC",
                (view.shape[1] - 60, 28),
                (0, 0, 255),
            )
        cv2.imshow(self.WINDOW_NAME, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("m"), ord("M")):
            self._toggle_mask()
        elif key in (ord("r"), ord("R")):
            try:
                self._toggle_recording(color_bgr)
            except (OSError, RuntimeError, ValueError, cv2.error) as exc:
                print(
                    f"[KinectViewer] RGB video recording failed: {exc}",
                    flush=True,
                )
        self.save_requested = self.is_save_key(key)
        try:
            visible = cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE)
        except cv2.error:
            visible = 0.0
        return key not in (ord("q"), 27) and visible >= 1.0

    def close(self):
        if self.closed:
            return
        self.closed = True
        summary = self.video_recorder.stop()
        if summary is not None:
            print(
                "[KinectViewer] RGB video saved while closing: "
                f"{summary['path']} ({summary['frames']} frames, "
                f"{summary['elapsed_s']:.1f}s)",
                flush=True,
            )
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
            cv2.waitKey(1)
        except cv2.error:
            pass


class EverythingSegmentationSession:
    """Propagate a clicked SAM3 mask and back-project only its current depth."""

    def __init__(
        self,
        *,
        worker,
        initial_mask,
        pointcloud_config,
        pointcloud_every,
        masked_cloud_fn,
        empty_cloud_factory,
        sam_hz,
        mask_max_age_s,
    ):
        self.worker = worker
        self.mask = np.asarray(np.asarray(initial_mask) > 0, dtype=np.uint8)
        self.pointcloud_config = pointcloud_config
        self.pointcloud_every = int(pointcloud_every)
        self.masked_cloud_fn = masked_cloud_fn
        self.empty_cloud_factory = empty_cloud_factory
        self.submit_interval = 1.0 / float(sam_hz)
        self.mask_max_age_s = float(mask_max_age_s)
        self.next_submit_at = 0.0
        self.first_live_frame_at = None
        self.latest_mask_at = None
        self.inference_ms = 0.0
        self.mask_results = 0
        self.submitted = 0
        self.replaced = 0

    def _consume_results(self, expected_shape, now):
        for message in self.worker.poll():
            kind = message.get("kind")
            if kind == "ready":
                print(
                    "[KinectPublisher] Everything SAM3 ready: "
                    f"load={message['load_ms']:.0f} ms "
                    f"GPU={message['gpu_allocated_mib']}/"
                    f"{message['gpu_reserved_mib']} MiB allocated/reserved",
                    flush=True,
                )
            elif kind == "fatal":
                details = message.get("traceback", message.get("message", "unknown error"))
                raise RuntimeError(f"Everything SAM3 worker failed:\n{details}")
            elif kind == "frame_error":
                print(
                    "[KinectPublisher] Everything SAM3 rejected frame "
                    f"{message.get('frame_id')}: {message.get('message')}",
                    flush=True,
                )
            elif kind == "mask":
                mask = np.asarray(np.asarray(message["mask"]) > 0, dtype=np.uint8)
                if mask.shape != expected_shape or not np.any(mask):
                    print(
                        "[KinectPublisher] Ignoring invalid Everything mask for "
                        f"frame {message.get('frame_id')}",
                        flush=True,
                    )
                    continue
                self.mask = mask
                self.latest_mask_at = float(message.get("captured_at", now))
                self.inference_ms = float(message["inference_ms"])
                self.mask_results += 1
                if self.mask_results % 10 == 0:
                    print(
                        "[KinectPublisher] Everything mask "
                        f"frame={message['frame_id']} pixels={message['pixels']} "
                        f"sam={self.inference_ms:.1f} ms method={message['method']}",
                        flush=True,
                    )

    def track(self, frame, captured_at):
        if self.first_live_frame_at is None:
            self.first_live_frame_at = captured_at
        self._consume_results(frame.depth_mm.shape, captured_at)
        if not self.worker.alive:
            raise RuntimeError("Everything SAM3 worker stopped unexpectedly")
        if self.mask.shape != frame.depth_mm.shape:
            raise ValueError("Everything initialization mask does not match live Kinect frames")

        if captured_at >= self.next_submit_at:
            self.next_submit_at = captured_at + self.submit_interval
            rgb = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2RGB)
            if self.worker.submit(frame.frame_id, captured_at, rgb):
                self.replaced += 1
            self.submitted += 1

        mask_timestamp = (
            self.latest_mask_at
            if self.latest_mask_at is not None
            else self.first_live_frame_at
        )
        segmentation_valid = (
            bool(np.any(self.mask))
            and 0 <= captured_at - mask_timestamp <= self.mask_max_age_s
        )
        update_cloud = (
            segmentation_valid
            and self.pointcloud_config is not None
            and (frame.frame_id - 1) % self.pointcloud_every == 0
        )
        observed = (
            self.masked_cloud_fn(frame, self.mask, self.pointcloud_config)
            if update_cloud
            else self.empty_cloud_factory()
        )
        return _TrackingResult(
            frame_id=frame.frame_id,
            pose=np.eye(4, dtype=np.float32),
            inference_ms=self.inference_ms,
            observed=observed,
            clouds_updated=update_cloud,
            segmentation_valid=segmentation_valid,
        )

    def close(self):
        self.worker.close()


class SynchronousEverythingSession:
    """Produce a same-frame tracker mask and model-free point cloud."""

    def __init__(
        self,
        *,
        tracker,
        initial_frame,
        initial_mask,
        pointcloud_config,
        pointcloud_every,
        masked_cloud_fn,
        empty_cloud_factory,
        max_consecutive_errors,
        tracker_name="tracker",
        mask_erosion_px=0,
        erode_mask_fn=None,
    ):
        self.tracker = tracker
        self.pointcloud_config = pointcloud_config
        self.pointcloud_every = int(pointcloud_every)
        self.masked_cloud_fn = masked_cloud_fn
        self.empty_cloud_factory = empty_cloud_factory
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.tracker_name = str(tracker_name)
        self.mask_erosion_px = int(mask_erosion_px)
        self.erode_mask_fn = erode_mask_fn
        if self.mask_erosion_px < 0:
            raise ValueError("Final mask erosion radius cannot be negative")
        if self.mask_erosion_px and self.erode_mask_fn is None:
            raise ValueError("Final mask erosion requires an erosion function")
        tracked_mask = tracker.initialize(initial_frame.color_bgr, initial_mask)
        self.raw_mask = np.ascontiguousarray(
            np.asarray(np.asarray(tracked_mask) > 0, dtype=np.uint8)
        )
        self.mask = self._postprocess_mask(self.raw_mask)
        self.refinement_box = getattr(tracker, "last_box", None)
        self.frames = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.inference_ms = 0.0

    def _postprocess_mask(self, mask):
        binary = np.asarray(np.asarray(mask) > 0, dtype=np.uint8)
        if self.mask_erosion_px:
            binary = np.asarray(
                self.erode_mask_fn(binary, self.mask_erosion_px),
                dtype=np.uint8,
            )
        if not np.any(binary):
            raise ValueError(
                f"{self.tracker_name} mask is empty after "
                f"{self.mask_erosion_px}px erosion"
            )
        return np.ascontiguousarray(binary)

    def track(self, frame):
        segmentation_valid = True
        try:
            tracked_mask, self.inference_ms = self.tracker.track(frame.color_bgr)
            raw_mask = np.ascontiguousarray(
                np.asarray(np.asarray(tracked_mask) > 0, dtype=np.uint8)
            )
            final_mask = self._postprocess_mask(raw_mask)
            self.raw_mask = raw_mask
            self.mask = final_mask
            self.consecutive_errors = 0
        except ValueError as exc:
            self.errors += 1
            self.consecutive_errors += 1
            segmentation_valid = False
            if self.errors == 1 or self.errors % 30 == 0:
                print(
                    f"[KinectPublisher] {self.tracker_name} mask rejected: {exc}",
                    flush=True,
                )
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise RuntimeError(
                    f"{self.tracker_name} produced invalid masks for "
                    f"{self.consecutive_errors} consecutive frames; rerun "
                    "--everything initialization and restart the publisher"
                ) from exc
        self.refinement_box = getattr(self.tracker, "last_box", None)
        self.frames += 1
        update_cloud = (
            segmentation_valid
            and self.pointcloud_config is not None
            and (frame.frame_id - 1) % self.pointcloud_every == 0
        )
        observed = (
            self.masked_cloud_fn(frame, self.mask, self.pointcloud_config)
            if update_cloud
            else self.empty_cloud_factory()
        )
        return _TrackingResult(
            frame_id=frame.frame_id,
            pose=np.eye(4, dtype=np.float32),
            inference_ms=self.inference_ms,
            observed=observed,
            clouds_updated=update_cloud,
            segmentation_valid=segmentation_valid,
        )

    def close(self):
        self.tracker.close()


# Retained for callers and tests written before the synchronous session was
# generalized for SAM2.
CutieEverythingSession = SynchronousEverythingSession


def load_calibrated_intrinsics(calibration_path):
    """Load the refined color-camera matrix used by hand-eye calibration."""
    path = Path(calibration_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Camera calibration JSON not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read camera calibration JSON: {path}") from exc
    if data.get("calibrated") is not True:
        raise ValueError(f"Camera calibration is not marked calibrated=true: {path}")

    intrinsics = np.asarray(data.get("camera_matrix"), dtype=np.float64)
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise ValueError(f"camera_matrix must be a finite 3x3 matrix: {path}")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError(f"camera_matrix focal lengths must be positive: {path}")
    if not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"camera_matrix last row must be [0, 0, 1]: {path}")
    return intrinsics.astype(np.float32)


def load_initialization(init_dir, RGBDFrame, *, intrinsics_override=None):
    init_dir = Path(init_dir)
    color = cv2.imread(str(init_dir / "rgb.png"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(init_dir / "depth.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(init_dir / "mask.png"), cv2.IMREAD_GRAYSCALE)
    camera_path = init_dir / "camera.json"
    if color is None or depth is None or mask is None:
        raise FileNotFoundError(
            f"Initialization requires rgb.png, depth.png and mask.png: {init_dir}"
        )
    if intrinsics_override is None:
        if not camera_path.is_file():
            raise FileNotFoundError(
                f"Initialization camera.json not found: {camera_path}"
            )
        intrinsics = np.asarray(
            json.loads(camera_path.read_text(encoding="utf-8"))["K"],
            dtype=np.float32,
        )
    else:
        intrinsics = np.asarray(intrinsics_override, dtype=np.float32)
    frame = RGBDFrame(color, depth, intrinsics).validated()
    if mask.shape != depth.shape or not np.any(mask):
        raise ValueError("Target initialization mask is empty or misaligned")
    return frame, np.asarray(mask > 0, dtype=np.uint8) * 255


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Azure Kinect + FoundationPose publisher for Wuji teleop"
    )
    parser.add_argument("--bind", default="tcp://127.0.0.1:6200")
    parser.add_argument("--foundation-root", type=Path, default=DEFAULT_FOUNDATION_ROOT)
    parser.add_argument("--config", type=Path)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--object", default=None)
    target.add_argument(
        "--everything",
        action="store_true",
        help="Publish the clicked target's segmented point cloud without a mesh or pose",
    )
    parser.add_argument(
        "--everything-tracker",
        choices=("cutie", "sam3"),
        default="cutie",
        help="Same-frame Cutie tracking or legacy low-rate SAM3 propagation",
    )
    parser.add_argument(
        "--sam2",
        action="store_true",
        help="Refine each Cutie mask with SAM2 using an expanded box prompt",
    )
    parser.add_argument("--init-dir", type=Path)
    parser.add_argument(
        "--camera-calibration",
        type=Path,
        default=DEFAULT_CAMERA_CALIBRATION,
        help=(
            "Calibration JSON whose camera_matrix is used for registration, "
            "tracking, point-cloud backprojection and publication"
        ),
    )
    parser.add_argument("--fps", type=int, choices=(5, 15, 30), default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--color-exposure-usec",
        type=int,
        help=(
            "Azure Kinect color exposure in microseconds (default: 10000); "
            "use 0 to restore automatic exposure"
        ),
        default=10000,
    )
    parser.add_argument(
        "--color-brightness",
        type=int,
        help=(
            "Use manual Azure Kinect color brightness; the camera-reported "
            "range and step are validated at startup"
        ),
        default=120,
    )
    parser.add_argument("--pointcloud-every", type=int, default=1)
    parser.add_argument("--pointcloud-max-points", type=int, default=8192)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--depth-png-compression", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-pointcloud", action="store_true")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Show the current RGB frame and Everything segmentation mask",
    )
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=DEFAULT_CAPTURE_DIR,
        help="Directory used by Enter-key RGB-D and point-cloud capture",
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=DEFAULT_VIDEO_DIR,
        help="Directory used by R-key raw RGB video recording",
    )
    parser.add_argument(
        "--connect-existing",
        action="store_true",
        help="Connect to an already-running FoundationPose service instead of starting Docker",
    )
    parser.add_argument("--fp-host", default="127.0.0.1")
    parser.add_argument("--fp-port", type=int)
    parser.add_argument("--sam-hz", type=float)
    parser.add_argument("--sam-window-seconds", type=float)
    parser.add_argument("--sam-min-mask-pixels", type=int)
    parser.add_argument("--sam-max-area-change", type=float)
    parser.add_argument("--cutie-weights", type=Path)
    parser.add_argument("--cutie-size", type=int, default=480)
    parser.add_argument("--cutie-mem-every", type=int, default=10)
    parser.add_argument("--cutie-min-mask-pixels", type=int)
    parser.add_argument("--cutie-max-area-change", type=float, default=3.0)
    parser.add_argument("--cutie-max-consecutive-errors", type=int, default=30)
    parser.add_argument(
        "--mask-erosion-px",
        type=int,
        help=(
            "Erode the final Cutie or Cutie+SAM2 mask by this image-space "
            "radius before display and point-cloud generation"
        ),
    )
    parser.add_argument(
        "--mask-dilation-px",
        type=int,
        default=0,
        help=(
            "Dilate the pre-erosion Cutie mask by this image-space radius "
            "for Enter-key point-cloud capture only"
        ),
    )
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--sam2-config")
    parser.add_argument("--sam2-box-padding", type=float, default=0.10)
    parser.add_argument("--sam2-crop-padding", type=float, default=0.50)
    parser.add_argument("--sam2-min-mask-pixels", type=int)
    parser.add_argument("--sam2-max-area-change", type=float, default=3.0)
    parser.add_argument("--sam2-max-consecutive-errors", type=int, default=30)
    args = parser.parse_args(argv)
    if args.sam2:
        if args.object is not None:
            parser.error("--sam2 cannot be combined with --object")
        if args.everything_tracker == "sam3":
            parser.error("--sam2 cannot be combined with --everything-tracker sam3")
        args.everything = True
    if args.warmup < 0 or args.pointcloud_every < 1 or args.pointcloud_max_points < 0:
        parser.error("warmup/max-points must be nonnegative and pointcloud-every positive")
    if args.color_exposure_usec is not None and args.color_exposure_usec < 0:
        parser.error("color-exposure-usec must be nonnegative")
    if args.color_exposure_usec == 0:
        args.color_exposure_usec = None
    if args.color_brightness is not None and args.color_brightness < 0:
        parser.error("color-brightness must be nonnegative")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be within [1, 100]")
    if not 0 <= args.depth_png_compression <= 9:
        parser.error("depth-png-compression must be within [0, 9]")
    if args.everything and args.connect_existing:
        parser.error("--everything does not use a FoundationPose service")
    if args.viewer and not args.everything:
        parser.error("--viewer requires --everything")
    if args.everything_tracker != "cutie" and not args.everything:
        parser.error("--everything-tracker requires --everything")
    if args.sam_hz is not None and not 0 < args.sam_hz <= 30:
        parser.error("sam-hz must be within (0, 30]")
    if args.sam_window_seconds is not None and args.sam_window_seconds <= 0:
        parser.error("sam-window-seconds must be positive")
    if args.sam_min_mask_pixels is not None and args.sam_min_mask_pixels < 1:
        parser.error("sam-min-mask-pixels must be positive")
    if args.sam_max_area_change is not None and args.sam_max_area_change <= 1:
        parser.error("sam-max-area-change must be greater than 1")
    if args.cutie_size == 0 or args.cutie_size < -1:
        parser.error("cutie-size must be -1 or positive")
    if args.cutie_mem_every < 1:
        parser.error("cutie-mem-every must be positive")
    if args.cutie_min_mask_pixels is not None and args.cutie_min_mask_pixels < 1:
        parser.error("cutie-min-mask-pixels must be positive")
    if args.cutie_max_area_change <= 1:
        parser.error("cutie-max-area-change must be greater than 1")
    if args.cutie_max_consecutive_errors < 1:
        parser.error("cutie-max-consecutive-errors must be positive")
    if args.mask_erosion_px is not None and args.mask_erosion_px < 0:
        parser.error("mask-erosion-px must be nonnegative")
    if args.mask_erosion_px is not None and (
        not args.everything or args.everything_tracker != "cutie"
    ):
        parser.error(
            "--mask-erosion-px requires --everything with the Cutie tracker"
        )
    if args.mask_dilation_px < 0:
        parser.error("mask-dilation-px must be nonnegative")
    if args.mask_dilation_px and (
        not args.everything
        or args.everything_tracker != "cutie"
        or not args.viewer
    ):
        parser.error(
            "--mask-dilation-px requires --everything --viewer with "
            "the Cutie tracker"
        )
    if args.sam2_min_mask_pixels is not None and args.sam2_min_mask_pixels < 1:
        parser.error("sam2-min-mask-pixels must be positive")
    if not 0 <= args.sam2_box_padding < args.sam2_crop_padding <= 1:
        parser.error(
            "SAM2 padding must satisfy 0 <= box-padding < crop-padding <= 1"
        )
    if args.sam2_max_area_change <= 1:
        parser.error("sam2-max-area-change must be greater than 1")
    if args.sam2_max_consecutive_errors < 1:
        parser.error("sam2-max-consecutive-errors must be positive")
    sam_overrides = (
        args.sam_hz,
        args.sam_window_seconds,
        args.sam_min_mask_pixels,
        args.sam_max_area_change,
    )
    if any(value is not None for value in sam_overrides) and (
        not args.everything or args.everything_tracker != "sam3"
    ):
        parser.error("SAM override options require --everything-tracker sam3")
    cutie_overrides = (
        args.cutie_weights,
        args.cutie_min_mask_pixels,
    )
    cutie_nondefaults = (
        args.cutie_size != 480
        or args.cutie_mem_every != 10
        or args.cutie_max_area_change != 3.0
        or args.cutie_max_consecutive_errors != 30
    )
    if (any(value is not None for value in cutie_overrides) or cutie_nondefaults) and (
        not args.everything or args.everything_tracker != "cutie"
    ):
        parser.error("Cutie options require --everything-tracker cutie")
    sam2_overrides = (
        args.sam2_checkpoint,
        args.sam2_config,
        args.sam2_min_mask_pixels,
    )
    sam2_nondefaults = (
        args.sam2_box_padding != 0.10
        or args.sam2_crop_padding != 0.50
        or args.sam2_max_area_change != 3.0
        or args.sam2_max_consecutive_errors != 30
    )
    if (any(value is not None for value in sam2_overrides) or sam2_nondefaults) and not args.sam2:
        parser.error("SAM2 options require --sam2")
    return args


def main():
    args = parse_args()
    foundation_root = args.foundation_root.resolve()
    for import_root in (foundation_root, foundation_root / "src"):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

    from foundationpose_kinect import (
        FoundationPoseService,
        FoundationPoseSession,
        PointCloud,
        ProjectConfig,
        RGBDFrame,
        masked_cloud,
    )
    from foundationpose_kinect.capture import save_rgbd_capture
    from foundationpose_kinect.geometry import erode_binary_mask
    from stream_client import azure_frames

    config_path = (
        args.config.resolve()
        if args.config
        else foundation_root / "configs" / "kinect_foundationpose.json"
    )
    project_config = ProjectConfig(config_path, foundation_root)
    object_spec = None if args.everything else project_config.object(args.object)
    runtime_intrinsics = load_calibrated_intrinsics(args.camera_calibration)
    calibration_path = args.camera_calibration.expanduser().resolve()
    calibration_metadata = json.loads(calibration_path.read_text(encoding="utf-8"))
    print(
        f"[KinectPublisher] Runtime intrinsics: {calibration_path}\n"
        f"[KinectPublisher] calibrated K = "
        f"fx={runtime_intrinsics[0, 0]:.2f} "
        f"fy={runtime_intrinsics[1, 1]:.2f} "
        f"cx={runtime_intrinsics[0, 2]:.2f} "
        f"cy={runtime_intrinsics[1, 2]:.2f}",
        flush=True,
    )
    init_dir = (
        args.init_dir.resolve()
        if args.init_dir
        else foundation_root / "foundationpose_cpp" / "test_data" / "live_init"
    )
    initial_frame, initial_mask = load_initialization(
        init_dir,
        RGBDFrame,
        intrinsics_override=runtime_intrinsics,
    )

    pointcloud_config = None
    if not args.no_pointcloud and args.pointcloud_max_points:
        pointcloud_config = replace(
            project_config.pointcloud_config(),
            max_points=args.pointcloud_max_points,
        )
        if args.everything and args.mask_erosion_px is not None:
            pointcloud_config = replace(pointcloud_config, mask_erosion_px=0)
        if args.everything:
            if args.mask_erosion_px is not None:
                print(
                    "[KinectPublisher] Final segmentation mask erosion: "
                    f"{args.mask_erosion_px} px before viewer and point cloud; "
                    "downstream cloud erosion disabled",
                    flush=True,
                )
            else:
                print(
                    "[KinectPublisher] Segmented-cloud mask erosion: "
                    f"{pointcloud_config.mask_erosion_px} px before "
                    f"sampling stride={pointcloud_config.stride}",
                    flush=True,
                )
        else:
            erosion_cells = (
                int(
                    np.ceil(
                        pointcloud_config.mask_erosion_px / pointcloud_config.stride
                    )
                )
                if pointcloud_config.mask_erosion_px
                else 0
            )
            print(
                "[KinectPublisher] Target-cloud mask erosion: "
                f"{pointcloud_config.mask_erosion_px} px requested; "
                f"approximately {erosion_cells * pointcloud_config.stride} px at "
                f"sampling stride={pointcloud_config.stride}",
                flush=True,
            )

    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.setsockopt(zmq.LINGER, 0)
    publisher.setsockopt(zmq.SNDHWM, 2)
    publisher.bind(args.bind)

    stopping = False

    def request_stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    sent = 0
    dropped = 0
    started = None
    stopped = None
    frames = None
    async_capture = None
    viewer = None
    encode_executor = None
    pending_encoded = None
    capture_executor = None
    pending_capture = None
    scheduled = 0

    def send_message(parts):
        nonlocal sent, dropped
        try:
            publisher.send_multipart(parts, flags=zmq.NOBLOCK)
            sent += 1
        except zmq.Again:
            dropped += 1

    try:
        with ExitStack() as stack:
            if args.everything:
                segmentation = project_config.data.get("segmentation", {})
                hybrid = project_config.data.get("hybrid", {})
                source_id = "azure-kinect:everything"
                min_mask_pixels = int(hybrid.get("min_mask_pixels", 500))
                if args.everything_tracker == "cutie":
                    from utils.cutie_tracker import CutieTracker

                    cutie_min_pixels = (
                        args.cutie_min_mask_pixels or min_mask_pixels
                    )
                    weights = (
                        args.cutie_weights.expanduser().resolve()
                        if args.cutie_weights
                        else foundation_root / "weights" / "cutie-base-mega.pth"
                    )
                    cutie_tracker = CutieTracker(
                        weights,
                        max_internal_size=args.cutie_size,
                        mem_every=args.cutie_mem_every,
                        min_mask_pixels=cutie_min_pixels,
                        max_area_change=args.cutie_max_area_change,
                    )
                    tracker = cutie_tracker
                    tracker_name = "Cutie"
                    max_consecutive_errors = args.cutie_max_consecutive_errors
                    tracking_label = "cutie"
                    mode_details = (
                        f"internal_short_edge={args.cutie_size} "
                        f"mem_every={args.cutie_mem_every}"
                    )
                    if args.mask_erosion_px is not None:
                        mode_details += (
                            f" final_mask_erosion={args.mask_erosion_px}px"
                        )
                    if args.sam2:
                        from utils.sam2_tracker import (
                            CutieSam2Tracker,
                            Sam2BoxRefiner,
                        )

                        sam2_min_pixels = (
                            args.sam2_min_mask_pixels or min_mask_pixels
                        )
                        checkpoint = (
                            args.sam2_checkpoint.expanduser().resolve()
                            if args.sam2_checkpoint
                            else foundation_root
                            / "weights"
                            / "sam2.1_hiera_tiny.pt"
                        )
                        model_config = (
                            args.sam2_config
                            or "configs/sam2.1/sam2.1_hiera_t.yaml"
                        )
                        sam2_refiner = Sam2BoxRefiner(
                            checkpoint,
                            model_config=model_config,
                            box_padding=args.sam2_box_padding,
                            crop_padding=args.sam2_crop_padding,
                            min_mask_pixels=sam2_min_pixels,
                            max_area_change=args.sam2_max_area_change,
                        )
                        tracker = CutieSam2Tracker(cutie_tracker, sam2_refiner)
                        tracker_name = "Cutie+SAM2"
                        max_consecutive_errors = (
                            args.sam2_max_consecutive_errors
                        )
                        tracking_label = "cutie+sam2"
                        mode_details += (
                            f" sam2_box_padding={args.sam2_box_padding:.2f} "
                            f"sam2_crop_padding={args.sam2_crop_padding:.2f}"
                        )
                    session = CutieEverythingSession(
                        tracker=tracker,
                        initial_frame=initial_frame,
                        initial_mask=initial_mask,
                        pointcloud_config=pointcloud_config,
                        pointcloud_every=args.pointcloud_every,
                        masked_cloud_fn=masked_cloud,
                        empty_cloud_factory=PointCloud,
                        max_consecutive_errors=max_consecutive_errors,
                        tracker_name=tracker_name,
                        mask_erosion_px=args.mask_erosion_px or 0,
                        erode_mask_fn=erode_binary_mask,
                    )
                    stack.callback(session.close)
                    encode_executor = stack.enter_context(
                        ThreadPoolExecutor(
                            max_workers=1,
                            thread_name_prefix="kinect-message-encoder",
                        )
                    )
                    print(
                        f"[KinectPublisher] Everything/{tracker_name} mode: no mesh "
                        f"and no object pose; {mode_details}",
                        flush=True,
                    )
                else:
                    from server.sam3_live_worker import LiveSamWorker

                    def project_path(value, default):
                        path = Path(value or default)
                        return path if path.is_absolute() else foundation_root / path

                    sam_hz = args.sam_hz or float(hybrid.get("sam_hz", 5.0))
                    sam_window = args.sam_window_seconds or float(
                        hybrid.get("window_seconds", 2.0)
                    )
                    min_mask_pixels = args.sam_min_mask_pixels or min_mask_pixels
                    max_area_change = args.sam_max_area_change or float(
                        hybrid.get("max_mask_area_change_ratio", 2.5)
                    )
                    if not 0 < sam_hz <= 30:
                        raise ValueError("Everything SAM rate must be within (0, 30]")
                    if sam_window <= 0 or min_mask_pixels < 1 or max_area_change <= 1:
                        raise ValueError("Invalid Everything SAM validation settings")
                    checkpoint = project_path(
                        segmentation.get("checkpoint"), "sam3/weights/sam3.pt"
                    )
                    bpe = project_path(
                        segmentation.get("bpe"),
                        "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
                    )
                    if not checkpoint.is_file():
                        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")
                    if not bpe.is_file():
                        raise FileNotFoundError(f"SAM3 BPE file not found: {bpe}")
                    worker = LiveSamWorker(
                        checkpoint=checkpoint,
                        bpe=bpe,
                        prompts=(),
                        threshold=float(segmentation.get("threshold", 0.5)),
                        sam_hz=sam_hz,
                        window_seconds=sam_window,
                        min_pixels=min_mask_pixels,
                        max_area_change=max_area_change,
                        seed_mask=initial_mask,
                    )
                    session = EverythingSegmentationSession(
                        worker=worker,
                        initial_mask=initial_mask,
                        pointcloud_config=pointcloud_config,
                        pointcloud_every=args.pointcloud_every,
                        masked_cloud_fn=masked_cloud,
                        empty_cloud_factory=PointCloud,
                        sam_hz=sam_hz,
                        mask_max_age_s=sam_window,
                    )
                    stack.callback(session.close)
                    tracking_label = "sam3"
                    print(
                        "[KinectPublisher] Everything/SAM3 mode: no mesh and no "
                        f"object pose; target={sam_hz:.2f} Hz "
                        f"window={sam_window:.2f}s",
                        flush=True,
                    )
            else:
                if args.connect_existing:
                    fp_host = args.fp_host
                    fp_port = args.fp_port or project_config.port
                else:
                    service = stack.enter_context(
                        FoundationPoseService(
                            foundation_root,
                            config=config_path,
                            object_name=object_spec.name,
                            port=args.fp_port,
                        )
                    )
                    fp_host = "127.0.0.1"
                    fp_port = service.port

                session = stack.enter_context(
                    FoundationPoseSession(
                        host=fp_host,
                        port=fp_port,
                        mesh_path=object_spec.mesh_path,
                        pointcloud_config=pointcloud_config,
                        pointcloud_every=args.pointcloud_every,
                    )
                )
                registered = session.register(initial_frame, initial_mask)
                source_id = f"azure-kinect:{object_spec.name}"
                tracking_label = "fp"
                print(
                    f"[KinectPublisher] Registered {object_spec.name}: "
                    f"frame={registered.frame_id}, "
                    f"inference={registered.inference_ms:.1f} ms",
                    flush=True,
                )

            if args.viewer:
                viewer = SegmentationViewer(
                    video_dir=args.video_dir,
                    video_fps=args.fps,
                )
                stack.callback(viewer.close)
                capture_root = args.capture_dir.expanduser().resolve()
                capture_root.mkdir(parents=True, exist_ok=True)
                capture_executor = stack.enter_context(
                    ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix="kinect-capture-writer",
                    )
                )
                print(
                    "[KinectPublisher] Enter saves synchronized RGB-D and "
                    "eroded/raw/dilated-mask PLY clouds "
                    f"(dilation={args.mask_dilation_px}px) to {capture_root}",
                    flush=True,
                )

            frames = azure_frames(
                fps=args.fps,
                warmup=args.warmup,
                color_exposure_usec=args.color_exposure_usec,
                color_brightness=args.color_brightness,
            )
            if args.everything and args.everything_tracker == "cutie":
                async_capture = LatestFrameSource(frames)
                frames = async_capture
            print(
                f"[KinectPublisher] Publishing {args.bind} at {args.fps} FPS; "
                f"pointcloud_every={args.pointcloud_every}",
                flush=True,
            )
            factory_intrinsics_logged = False
            for captured in frames:
                if stopping:
                    break
                if pending_capture is not None and pending_capture.done():
                    saved = pending_capture.result()
                    print(
                        "[KinectPublisher] Capture saved: "
                        f"{saved['output']} "
                        f"(eroded={saved['masked_points']:,}, "
                        f"raw={saved['raw_mask_points']:,}, "
                        f"dilated={saved['dilated_mask_points']:,} points)",
                        flush=True,
                    )
                    pending_capture = None
                if async_capture is not None:
                    color_bgr = captured.color_bgr
                    depth_mm = captured.depth_mm
                    depth_scale = captured.depth_scale
                    factory_intrinsics = captured.intrinsics
                    capture_monotonic_ns = captured.monotonic_ns
                    capture_wall_ns = captured.wall_ns
                    frame_id = captured.frame_id
                else:
                    color_bgr, depth_mm, depth_scale, factory_intrinsics = captured
                    capture_monotonic_ns = time.monotonic_ns()
                    capture_wall_ns = time.time_ns()
                    frame_id = sent + dropped + 1
                if started is None:
                    started = time.monotonic()
                if not factory_intrinsics_logged:
                    factory_intrinsics = np.asarray(factory_intrinsics, dtype=np.float32)
                    print(
                        "[KinectPublisher] Azure factory K replaced for runtime: "
                        f"fx={factory_intrinsics[0, 0]:.2f} "
                        f"fy={factory_intrinsics[1, 1]:.2f} "
                        f"cx={factory_intrinsics[0, 2]:.2f} "
                        f"cy={factory_intrinsics[1, 2]:.2f}",
                        flush=True,
                    )
                    factory_intrinsics_logged = True
                live_frame = RGBDFrame(
                    color_bgr,
                    depth_mm,
                    runtime_intrinsics,
                    frame_id=frame_id,
                )
                result = (
                    session.track(live_frame, capture_monotonic_ns * 1e-9)
                    if args.everything and args.everything_tracker == "sam3"
                    else session.track(live_frame)
                )
                pose_valid = not args.everything
                segmentation_valid = (
                    result.segmentation_valid if args.everything else False
                )
                if viewer is not None:
                    display_mask = (
                        session.mask
                        if segmentation_valid
                        else np.zeros(depth_mm.shape, dtype=np.uint8)
                    )
                    if not viewer.show(
                        color_bgr,
                        display_mask,
                        valid=segmentation_valid,
                        box=(
                            session.refinement_box
                            if args.everything and args.sam2
                            else None
                        ),
                    ):
                        stopping = True
                    if viewer.save_requested:
                        if not segmentation_valid:
                            print(
                                "[KinectPublisher] Capture ignored: the current "
                                "segmentation mask is invalid",
                                flush=True,
                            )
                        elif pending_capture is not None:
                            print(
                                "[KinectPublisher] Capture ignored: the previous "
                                "capture is still being saved",
                                flush=True,
                            )
                        else:
                            captured_at = datetime.fromtimestamp(
                                capture_wall_ns / 1_000_000_000
                            ).astimezone()
                            capture_name = (
                                captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                + f"_frame_{frame_id:06d}"
                            )
                            output = capture_root / capture_name
                            pending_capture = capture_executor.submit(
                                save_rgbd_capture,
                                output,
                                color_bgr.copy(),
                                depth_mm.copy(),
                                float(depth_scale),
                                runtime_intrinsics.copy(),
                                captured_at=captured_at.isoformat(
                                    timespec="milliseconds"
                                ),
                                mask=display_mask.copy(),
                                raw_mask=np.asarray(
                                    getattr(session, "raw_mask", display_mask),
                                    dtype=np.uint8,
                                ).copy(),
                                mask_dilation_px=args.mask_dilation_px,
                                save_scene_cloud=False,
                                frame_id=frame_id,
                                factory_intrinsics=np.asarray(
                                    factory_intrinsics,
                                    dtype=np.float64,
                                ).copy(),
                                calibration=calibration_metadata,
                                segmentation_valid=segmentation_valid,
                            )
                            print(
                                f"[KinectPublisher] Saving capture to {output} ...",
                                flush=True,
                            )
                encode_options = dict(
                    frame_id=result.frame_id,
                    color_bgr=color_bgr,
                    depth_mm=depth_mm,
                    intrinsics=runtime_intrinsics,
                    object_pose_camera=result.pose,
                    capture_monotonic_ns=capture_monotonic_ns,
                    capture_wall_ns=capture_wall_ns,
                    inference_ms=result.inference_ms,
                    pose_valid=pose_valid,
                    sample_valid=pose_valid or segmentation_valid,
                    segmentation_valid=segmentation_valid,
                    tracking_mode=(
                        "everything" if args.everything else "foundationpose"
                    ),
                    clouds_updated=result.clouds_updated,
                    pointcloud_xyz=result.observed.points,
                    pointcloud_rgb=result.observed.colors_rgb,
                    source_id=source_id,
                    jpeg_quality=args.jpeg_quality,
                    depth_png_compression=args.depth_png_compression,
                )
                published_inference_ms = result.inference_ms
                published_cloud = len(result.observed.points)
                if encode_executor is None:
                    send_message(encode_kinect_message(**encode_options))
                else:
                    next_encoded = encode_executor.submit(
                        encode_kinect_message, **encode_options
                    )
                    scheduled += 1
                    if pending_encoded is not None:
                        previous_future, published_inference_ms, published_cloud = (
                            pending_encoded
                        )
                        send_message(previous_future.result())
                    else:
                        published_inference_ms = result.inference_ms
                        published_cloud = len(result.observed.points)
                    pending_encoded = (
                        next_encoded,
                        result.inference_ms,
                        len(result.observed.points),
                    )

                if sent and sent % 30 == 0:
                    elapsed = time.monotonic() - started
                    capture_drop_text = (
                        f" camera_dropped={async_capture.dropped}"
                        if async_capture is not None
                        else ""
                    )
                    print(
                        f"[KinectPublisher] sent={sent} dropped={dropped} "
                        f"rate={sent / elapsed:.1f} Hz "
                        f"{tracking_label}={published_inference_ms:.1f} ms "
                        f"cloud={published_cloud}{capture_drop_text}",
                        flush=True,
                    )
                reached_limit = (
                    scheduled >= args.max_frames
                    if encode_executor is not None
                    else sent >= args.max_frames
                )
                if args.max_frames and reached_limit:
                    break
            if pending_encoded is not None:
                send_message(pending_encoded[0].result())
            if pending_capture is not None:
                saved = pending_capture.result()
                print(
                    "[KinectPublisher] Capture saved: "
                    f"{saved['output']} "
                    f"(eroded={saved['masked_points']:,}, "
                    f"raw={saved['raw_mask_points']:,}, "
                    f"dilated={saved['dilated_mask_points']:,} points)",
                    flush=True,
                )
                pending_capture = None
            stopped = time.monotonic()
            if async_capture is not None:
                async_capture.close()
    finally:
        if frames is not None:
            frames.close()
        publisher.close(0)
        context.term()
        capture_drop_text = (
            f", camera_dropped={async_capture.dropped}"
            if async_capture is not None
            else ""
        )
        rate_text = (
            f", rate={sent / ((stopped or time.monotonic()) - started):.1f} Hz"
            if sent and started is not None
            else ""
        )
        print(
            f"[KinectPublisher] Stopped: sent={sent}, dropped={dropped}"
            f"{capture_drop_text}{rate_text}",
            flush=True,
        )


if __name__ == "__main__":
    main()
