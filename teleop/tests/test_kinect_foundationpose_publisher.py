import json
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from scripts.kinect_foundationpose_publisher import (
    CutieEverythingSession,
    EverythingSegmentationSession,
    LatestFrameSource,
    RgbVideoRecorder,
    SegmentationViewer,
    load_calibrated_intrinsics,
    load_initialization,
    parse_args,
    render_segmentation_view,
)


class _Frame:
    def __init__(self, color_bgr, depth_mm, intrinsics):
        self.color_bgr = color_bgr
        self.depth_mm = depth_mm
        self.intrinsics = intrinsics

    def validated(self):
        return self


class KinectPublisherIntrinsicsTest(unittest.TestCase):
    def test_loads_refined_intrinsics_from_hand_eye_json(self):
        matrix = np.array(
            [[613.7, 0.0, 644.9], [0.0, 617.2, 368.9], [0.0, 0.0, 1.0]]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps({"calibrated": True, "camera_matrix": matrix.tolist()}),
                encoding="utf-8",
            )
            actual = load_calibrated_intrinsics(path)

        np.testing.assert_allclose(actual, matrix, rtol=0.0, atol=1e-4)
        self.assertEqual(actual.dtype, np.float32)

    def test_initialization_override_does_not_use_saved_camera_json(self):
        override = np.array(
            [[610.0, 0.0, 640.0], [0.0, 611.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        with tempfile.TemporaryDirectory() as directory:
            init_dir = Path(directory)
            cv2.imwrite(str(init_dir / "rgb.png"), np.zeros((4, 6, 3), np.uint8))
            cv2.imwrite(str(init_dir / "depth.png"), np.ones((4, 6), np.uint16))
            cv2.imwrite(str(init_dir / "mask.png"), np.ones((4, 6), np.uint8))
            (init_dir / "camera.json").write_text(
                json.dumps({"K": np.eye(3).tolist()}), encoding="utf-8"
            )

            frame, mask = load_initialization(
                init_dir, _Frame, intrinsics_override=override
            )

        np.testing.assert_array_equal(frame.intrinsics, override)
        self.assertTrue(np.all(mask == 255))

    def test_rejects_uncalibrated_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            path.write_text(
                json.dumps({"calibrated": False, "camera_matrix": np.eye(3).tolist()}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "calibrated=true"):
                load_calibrated_intrinsics(path)


class _Cloud:
    def __init__(self, count=0):
        self.points = np.zeros((count, 3), dtype=np.float32)
        self.colors_rgb = np.zeros((count, 3), dtype=np.uint8)


class _Worker:
    alive = True

    def __init__(self, messages=()):
        self.messages = list(messages)
        self.requests = []
        self.closed = False

    def poll(self):
        messages, self.messages = self.messages, []
        return messages

    def submit(self, frame_id, captured_at, rgb):
        self.requests.append((frame_id, captured_at, rgb.copy()))
        return False

    def close(self):
        self.closed = True


class _CutieTracker:
    def __init__(self, mask, error=None):
        self.mask = mask
        self.error = error
        self.initialized = False
        self.closed = False

    def initialize(self, _color, initial_mask):
        self.initialized = True
        return np.asarray(initial_mask > 0, dtype=np.uint8)

    def track(self, _color):
        if self.error is not None:
            raise ValueError(self.error)
        return self.mask.copy(), 7.5

    def close(self):
        self.closed = True


class EverythingSegmentationTest(unittest.TestCase):
    class _VideoWriter:
        def __init__(self, path, fourcc, fps, frame_size):
            self.path = Path(path)
            self.fourcc = fourcc
            self.fps = fps
            self.frame_size = frame_size
            self.frames = []
            self.released = False

        def isOpened(self):
            return True

        def write(self, frame):
            self.frames.append(frame.copy())

        def release(self):
            self.released = True

    def test_viewer_recognizes_both_enter_key_codes(self):
        self.assertTrue(SegmentationViewer.is_save_key(10))
        self.assertTrue(SegmentationViewer.is_save_key(13))
        self.assertFalse(SegmentationViewer.is_save_key(ord("q")))

    def test_viewer_render_preserves_image_and_highlights_mask(self):
        color = np.full((10, 12, 3), 100, dtype=np.uint8)
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[2:8, 3:9] = 1

        view = render_segmentation_view(
            color, mask, valid=True, panel_size=(12, 10)
        )

        self.assertEqual(view.shape, (10, 24, 3))
        np.testing.assert_array_equal(view[:, :12], color)
        np.testing.assert_array_equal(view[5, 12 + 6], color[5, 6])
        np.testing.assert_array_equal(view[0, 12], np.full(3, 20, dtype=np.uint8))

    def test_viewer_draws_cutie_guide_box_on_rgb_panel(self):
        color = np.full((10, 12, 3), 100, dtype=np.uint8)
        mask = np.zeros((10, 12), dtype=np.uint8)
        mask[3:7, 4:8] = 1

        view = render_segmentation_view(
            color,
            mask,
            valid=True,
            box=np.array([2, 1, 9, 8], dtype=np.float32),
            panel_size=(12, 10),
        )

        np.testing.assert_array_equal(view[1, 2], np.array([0, 255, 255]))
        np.testing.assert_array_equal(view[3, 12 + 4], np.array([0, 255, 0]))

    def test_viewer_can_hide_mask_and_keep_full_rgb_panel(self):
        color = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
        mask = np.ones((10, 12), dtype=np.uint8)

        view = render_segmentation_view(
            color,
            mask,
            valid=True,
            panel_size=(12, 10),
            show_mask=False,
        )

        self.assertEqual(view.shape, color.shape)
        np.testing.assert_array_equal(view, color)

    def test_mask_control_switches_between_rgb_and_split_aspect_ratios(self):
        viewer = SegmentationViewer.__new__(SegmentationViewer)
        viewer.closed = False
        viewer.mask_visible = True

        with patch.object(viewer, "_window_width", return_value=1280), patch(
            "scripts.kinect_foundationpose_publisher.cv2.resizeWindow"
        ) as resize:
            viewer._on_mask_trackbar(0)
            self.assertFalse(viewer.mask_visible)
            resize.assert_called_once_with(viewer.WINDOW_NAME, 1280, 720)

            resize.reset_mock()
            viewer._on_mask_trackbar(1)
            self.assertTrue(viewer.mask_visible)
            resize.assert_called_once_with(viewer.WINDOW_NAME, 1280, 360)

    def test_rgb_video_recorder_writes_raw_frames_and_releases_writer(self):
        writers = []

        def factory(*args):
            writer = self._VideoWriter(*args)
            writers.append(writer)
            return writer

        with tempfile.TemporaryDirectory() as directory:
            recorder = RgbVideoRecorder(directory, 30, writer_factory=factory)
            frame = np.full((4, 6, 3), [10, 20, 30], dtype=np.uint8)
            path = recorder.start(
                frame,
                captured_at=datetime(2026, 9, 19, 17, 30, 0),
            )
            recorder.write(frame)
            summary = recorder.stop()

        self.assertEqual(path.name, "kinect_rgb_20260919_173000_000000.mp4")
        self.assertEqual(summary["path"], path)
        self.assertEqual(summary["frames"], 1)
        self.assertEqual(writers[0].fps, 30)
        self.assertEqual(writers[0].frame_size, (6, 4))
        np.testing.assert_array_equal(writers[0].frames[0], frame)
        self.assertTrue(writers[0].released)

    def test_viewer_requires_everything_mode(self):
        with self.assertRaises(SystemExit):
            parse_args(["--viewer"])
        self.assertTrue(parse_args(["--everything", "--viewer"]).viewer)

    def test_latest_frame_source_discards_backlog_and_preserves_capture_ids(self):
        class Frames:
            def __init__(self):
                self.closed = False

            def __iter__(self):
                for value in range(3):
                    color = np.full((2, 3, 3), value, dtype=np.uint8)
                    yield color, np.ones((2, 3), np.uint16), 0.001, np.eye(3)

            def close(self):
                self.closed = True

        frames = Frames()
        source = LatestFrameSource(frames)
        self.assertTrue(source.finished.wait(timeout=1.0))
        captured = next(source)
        source.close()

        self.assertEqual(captured.frame_id, 3)
        self.assertEqual(int(captured.color_bgr[0, 0, 0]), 2)
        self.assertEqual(source.captured, 3)
        self.assertEqual(source.dropped, 2)
        self.assertTrue(frames.closed)

    def test_everything_cli_is_model_independent(self):
        args = parse_args(["--everything"])
        self.assertTrue(args.everything)
        self.assertIsNone(args.object)
        self.assertEqual(args.everything_tracker, "cutie")

    def test_color_camera_controls_are_optional_and_validated(self):
        defaults = parse_args(["--everything"])
        self.assertEqual(defaults.color_exposure_usec, 10000)
        self.assertEqual(defaults.color_brightness, 120)

        args = parse_args(
            [
                "--everything",
                "--color-exposure-usec",
                "8000",
                "--color-brightness",
                "100",
            ]
        )
        self.assertEqual(args.color_exposure_usec, 8000)
        self.assertEqual(args.color_brightness, 100)

        automatic = parse_args(
            ["--everything", "--color-exposure-usec", "0"]
        )
        self.assertIsNone(automatic.color_exposure_usec)
        with self.assertRaises(SystemExit):
            parse_args(["--everything", "--color-exposure-usec", "-1"])
        with self.assertRaises(SystemExit):
            parse_args(["--everything", "--color-brightness", "-1"])

    def test_sam2_shortcut_enables_everything_at_native_15_fps(self):
        args = parse_args(["--sam2", "--fps", "15"])
        self.assertTrue(args.everything)
        self.assertTrue(args.sam2)
        self.assertIsNone(args.object)
        self.assertEqual(args.everything_tracker, "cutie")
        self.assertEqual(args.fps, 15)

    def test_sam2_options_require_sam2_mode(self):
        with self.assertRaises(SystemExit):
            parse_args(["--sam2-checkpoint", "tiny.pt"])
        with self.assertRaises(SystemExit):
            parse_args(["--object", "apple_10cm", "--sam2"])
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--sam2",
                    "--sam2-box-padding",
                    "0.5",
                    "--sam2-crop-padding",
                    "0.4",
                ]
            )

    def test_final_mask_erosion_cli_requires_cutie_everything_mode(self):
        args = parse_args(["--everything", "--mask-erosion-px", "5"])
        self.assertEqual(args.mask_erosion_px, 5)

        with self.assertRaises(SystemExit):
            parse_args(["--mask-erosion-px", "2"])
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--everything",
                    "--everything-tracker",
                    "sam3",
                    "--mask-erosion-px",
                    "2",
                ]
            )
        with self.assertRaises(SystemExit):
            parse_args(["--everything", "--mask-erosion-px", "-1"])

    def test_mask_dilation_is_capture_only_and_requires_cutie_viewer(self):
        args = parse_args(
            [
                "--everything",
                "--viewer",
                "--mask-dilation-px",
                "5",
            ]
        )
        self.assertEqual(args.mask_dilation_px, 5)

        with self.assertRaises(SystemExit):
            parse_args(["--everything", "--mask-dilation-px", "2"])
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--everything",
                    "--viewer",
                    "--everything-tracker",
                    "sam3",
                    "--mask-dilation-px",
                    "2",
                ]
            )
        with self.assertRaises(SystemExit):
            parse_args(
                [
                    "--everything",
                    "--viewer",
                    "--mask-dilation-px",
                    "-1",
                ]
            )

    def test_cutie_uses_same_frame_mask_for_cloud(self):
        mask = np.zeros((4, 6), dtype=np.uint8)
        mask[1:3, 2:5] = 1
        tracker = _CutieTracker(mask)
        frame = _Frame(
            np.zeros((4, 6, 3), dtype=np.uint8),
            np.ones((4, 6), dtype=np.uint16),
            np.eye(3, dtype=np.float32),
        )
        frame.frame_id = 1
        seen_masks = []

        def make_cloud(_frame, tracked_mask, _config):
            seen_masks.append(tracked_mask.copy())
            return _Cloud(6)

        session = CutieEverythingSession(
            tracker=tracker,
            initial_frame=frame,
            initial_mask=np.ones((4, 6), dtype=np.uint8),
            pointcloud_config=object(),
            pointcloud_every=1,
            masked_cloud_fn=make_cloud,
            empty_cloud_factory=_Cloud,
            max_consecutive_errors=30,
        )
        result = session.track(frame)

        self.assertTrue(tracker.initialized)
        np.testing.assert_array_equal(seen_masks[0], mask)
        self.assertTrue(result.segmentation_valid)
        self.assertEqual(result.inference_ms, 7.5)
        self.assertEqual(len(result.observed.points), 6)

    def test_cutie_erodes_final_mask_before_viewer_and_cloud(self):
        raw_mask = np.zeros((9, 9), dtype=np.uint8)
        raw_mask[1:8, 1:8] = 1
        expected = np.zeros_like(raw_mask)
        expected[3:6, 3:6] = 1
        tracker = _CutieTracker(raw_mask)
        frame = _Frame(
            np.zeros((9, 9, 3), dtype=np.uint8),
            np.ones((9, 9), dtype=np.uint16),
            np.eye(3, dtype=np.float32),
        )
        frame.frame_id = 1
        seen_masks = []
        erosion_radii = []

        def erode(mask, radius):
            erosion_radii.append(radius)
            if np.array_equal(mask, raw_mask):
                return expected
            return mask

        def make_cloud(_frame, tracked_mask, _config):
            seen_masks.append(tracked_mask.copy())
            return _Cloud(9)

        session = CutieEverythingSession(
            tracker=tracker,
            initial_frame=frame,
            initial_mask=np.ones((9, 9), dtype=np.uint8),
            pointcloud_config=object(),
            pointcloud_every=1,
            masked_cloud_fn=make_cloud,
            empty_cloud_factory=_Cloud,
            max_consecutive_errors=30,
            mask_erosion_px=2,
            erode_mask_fn=erode,
        )
        result = session.track(frame)

        np.testing.assert_array_equal(session.raw_mask, raw_mask)
        np.testing.assert_array_equal(session.mask, expected)
        np.testing.assert_array_equal(seen_masks[0], expected)
        self.assertEqual(erosion_radii, [2, 2])
        self.assertTrue(result.segmentation_valid)

    def test_cutie_stops_after_consecutive_invalid_masks(self):
        tracker = _CutieTracker(np.ones((4, 6), dtype=np.uint8), error="lost target")
        frame = _Frame(
            np.zeros((4, 6, 3), dtype=np.uint8),
            np.ones((4, 6), dtype=np.uint16),
            np.eye(3, dtype=np.float32),
        )
        frame.frame_id = 1
        session = CutieEverythingSession(
            tracker=tracker,
            initial_frame=frame,
            initial_mask=np.ones((4, 6), dtype=np.uint8),
            pointcloud_config=object(),
            pointcloud_every=1,
            masked_cloud_fn=lambda *_args: _Cloud(6),
            empty_cloud_factory=_Cloud,
            max_consecutive_errors=2,
        )

        first = session.track(frame)
        self.assertFalse(first.segmentation_valid)
        self.assertFalse(first.clouds_updated)
        self.assertEqual(len(first.observed.points), 0)
        with self.assertRaisesRegex(RuntimeError, "2 consecutive frames"):
            session.track(frame)

    def test_mask_result_drives_cloud_without_a_pose(self):
        updated_mask = np.zeros((4, 6), dtype=np.uint8)
        updated_mask[1:3, 2:4] = 1
        worker = _Worker(
            [
                {
                    "kind": "mask",
                    "frame_id": 3,
                    "mask": updated_mask,
                    "inference_ms": 12.5,
                    "pixels": 4,
                    "method": "point-propagation",
                }
            ]
        )
        seen_masks = []

        def make_cloud(_frame, mask, _config):
            seen_masks.append(mask.copy())
            return _Cloud(4)

        session = EverythingSegmentationSession(
            worker=worker,
            initial_mask=np.ones((4, 6), dtype=np.uint8),
            pointcloud_config=object(),
            pointcloud_every=1,
            masked_cloud_fn=make_cloud,
            empty_cloud_factory=_Cloud,
            sam_hz=5,
            mask_max_age_s=2.0,
        )
        frame = _Frame(
            np.zeros((4, 6, 3), dtype=np.uint8),
            np.ones((4, 6), dtype=np.uint16),
            np.eye(3, dtype=np.float32),
        )
        frame.frame_id = 1
        result = session.track(frame, captured_at=10.0)

        np.testing.assert_array_equal(seen_masks[0], updated_mask)
        np.testing.assert_array_equal(result.pose, np.eye(4))
        self.assertTrue(result.segmentation_valid)
        self.assertEqual(result.inference_ms, 12.5)
        self.assertEqual(len(worker.requests), 1)
        np.testing.assert_array_equal(worker.requests[0][2], frame.color_bgr[..., ::-1])

        frame.frame_id = 2
        stale = session.track(frame, captured_at=12.1)
        self.assertFalse(stale.segmentation_valid)
        self.assertFalse(stale.clouds_updated)
        self.assertEqual(len(stale.observed.points), 0)


if __name__ == "__main__":
    unittest.main()
