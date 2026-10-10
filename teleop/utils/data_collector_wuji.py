"""Incremental HDF5 writer for Wuji teleoperation recordings."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time

import h5py
import numpy as np


class DataCollectorWuji:
    """Append synchronized timesteps directly to a recoverable HDF5 file."""

    def __init__(
        self,
        save_dir: str = "data_records_wuji",
        *,
        hand_model: str,
        hand_joint_names,
        hand_urdf: str,
        flange_thickness: float,
        camera_mode: str,
        camera_names,
        distance_keypoint_names,
        distance_T_base_camera=None,
        flush_every_frames: int = 15,
        arm_metadata=None,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.recording = False
        self.start_time = 0.0
        self.task_name = "default"
        self.input_device = "unknown"
        self.arm_metadata = dict(arm_metadata or {})
        self.hand_model = hand_model
        self.hand_joint_names = tuple(hand_joint_names)
        self.hand_urdf = hand_urdf
        self.flange_thickness = float(flange_thickness)
        self.camera_mode = str(camera_mode)
        self.camera_names = tuple(camera_names)
        self.distance_keypoint_names = tuple(distance_keypoint_names)
        self.distance_T_base_camera = None
        if distance_T_base_camera is not None:
            from utils.kinect_hand_eye import validate_T_base_camera

            self.distance_T_base_camera = validate_T_base_camera(
                distance_T_base_camera
            )
        if flush_every_frames < 1:
            raise ValueError("flush_every_frames must be positive")
        self.flush_every_frames = int(flush_every_frames)
        self.frame_count = 0
        self._file = None
        self._datasets = {}
        self._final_path = None
        self._inprogress_path = None
        self._lock = threading.Lock()

    @property
    def filepath(self):
        return self._final_path

    @property
    def inprogress_filepath(self):
        return self._inprogress_path

    def _write_metadata(self, h5_file, created_at):
        h5_file.attrs["task_name"] = self.task_name
        h5_file.attrs["input_device"] = self.input_device
        h5_file.attrs["created_at"] = created_at
        h5_file.attrs["recording_complete"] = False
        h5_file.attrs["writer"] = "streaming-v1"
        h5_file.attrs["arm_metadata"] = json.dumps(self.arm_metadata)
        h5_file.attrs["hand_model"] = self.hand_model
        h5_file.attrs["hand_dof"] = len(self.hand_joint_names)
        h5_file.attrs["hand_joint_names"] = json.dumps(self.hand_joint_names)
        h5_file.attrs["hand_joint_units"] = "radian"
        h5_file.attrs["hand_urdf"] = self.hand_urdf
        h5_file.attrs["flange_thickness_m"] = self.flange_thickness
        h5_file.attrs["camera_mode"] = self.camera_mode
        h5_file.attrs["camera_names"] = json.dumps(self.camera_names)
        h5_file.attrs["distance_keypoint_names"] = json.dumps(
            self.distance_keypoint_names
        )
        h5_file.attrs["distance_coordinate_frame"] = "base_link"
        h5_file.attrs["distance_units"] = "meter"
        h5_file.attrs["distance_metric"] = "nearest_euclidean"
        h5_file.attrs["distance_source"] = "kinect_observed_target_pointcloud"
        h5_file.attrs["distance_invalid_value"] = "NaN"
        if "KINECT" in self.camera_names:
            h5_file.attrs["kinect_depth_units"] = "millimeter"
            h5_file.attrs["kinect_pose_convention"] = "T_camera_object"
            h5_file.attrs["kinect_pointcloud_coordinate_frame"] = "camera"
            h5_file.attrs["kinect_timestamp_clock"] = "host_monotonic_ns"
            h5_file.attrs["kinect_hand_eye_convention"] = "T_base_camera"
            if self.distance_T_base_camera is not None:
                h5_file.attrs["kinect_T_base_camera"] = self.distance_T_base_camera

    def start_recording(self, task_name: str, input_device: str = "unknown"):
        """Open a new append-only recording, leaving an in-progress crash artifact."""
        with self._lock:
            if self.recording:
                raise RuntimeError("A Wuji recording is already active")
            self.task_name = task_name if task_name else "default"
            if input_device not in {"vr", "glove", "unknown"}:
                raise ValueError(f"Unsupported input device: {input_device}")
            self.input_device = input_device
            created_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{self.task_name}_{created_at}.h5"
            self._final_path = self.save_dir / filename
            self._inprogress_path = self.save_dir / f"{filename}.inprogress"
            self._file = h5py.File(self._inprogress_path, "w", libver="latest")
            self._datasets = {}
            self.frame_count = 0
            self.start_time = time.time()
            self._write_metadata(self._file, created_at)
            self._file.flush()
            self.recording = True
        print(
            f"[DataCollector] Recording STARTED: {self._inprogress_path}",
            flush=True,
        )

    def stop_recording(self):
        """Flush, close and atomically promote the in-progress file."""
        with self._lock:
            if not self.recording:
                return
            self.recording = False
            duration = time.time() - self.start_time
            frames = self.frame_count
            h5_file = self._file
            final_path = self._final_path
            inprogress_path = self._inprogress_path
            try:
                h5_file.attrs["total_frames"] = frames
                h5_file.attrs["duration_s"] = duration
                h5_file.attrs["avg_fps"] = frames / duration if duration > 0 else 0.0
                h5_file.attrs["recording_complete"] = True
                h5_file.flush()
            finally:
                h5_file.close()
                self._file = None
                self._datasets = {}
            os.replace(inprogress_path, final_path)

        avg_fps = frames / duration if duration > 0 else 0.0
        print("-" * 40)
        print(f"[DataCollector] Saved {final_path}")
        print(f"  Total Duration : {duration:.2f} s")
        print(f"  Total Frames   : {frames}")
        print(f"  Average FPS    : {avg_fps:.2f} Hz (Target: 15.0 Hz)")
        print("-" * 40)

    @staticmethod
    def _flatten(data, prefix=""):
        leaves = {}
        for key, value in data.items():
            path = f"{prefix}/{key}" if prefix else str(key)
            if isinstance(value, dict):
                leaves.update(DataCollectorWuji._flatten(value, path))
            else:
                array = np.asarray(value)
                if array.dtype.kind in {"O", "U"}:
                    raise TypeError(f"Unsupported HDF5 value at {path}: {array.dtype}")
                leaves[path] = array
        return leaves

    @staticmethod
    def _dataset_options(path, array):
        options = {
            "shape": (0,) + array.shape,
            "maxshape": (None,) + array.shape,
            "dtype": array.dtype,
            "chunks": (1,) + array.shape if array.shape else (256,),
        }
        # Images and fixed point clouds dominate the file; LZF is fast enough for
        # realtime append and avoids an optional compression dependency.
        if array.ndim >= 2 and array.size >= 1024:
            options["compression"] = "lzf"
            options["shuffle"] = True
        return options

    def _create_dataset(self, path, array):
        parent_path, _, name = path.rpartition("/")
        parent = self._file.require_group(parent_path) if parent_path else self._file
        options = self._dataset_options(path, array)
        dataset = parent.create_dataset(name, **options)
        if self.frame_count:
            dataset.resize((self.frame_count,) + array.shape)
        self._datasets[path] = dataset
        return dataset

    def add_frame(self, frame_data: dict):
        """Append one complete synchronized timestep to disk."""
        with self._lock:
            if not self.recording:
                return
            leaves = self._flatten(frame_data)
            for path, array in leaves.items():
                dataset = self._datasets.get(path)
                if dataset is None:
                    dataset = self._create_dataset(path, array)
                if dataset.shape[1:] != array.shape:
                    raise ValueError(
                        f"HDF5 field shape changed at {path}: "
                        f"{dataset.shape[1:]} -> {array.shape}"
                    )
                if dataset.dtype != array.dtype:
                    raise ValueError(
                        f"HDF5 field dtype changed at {path}: "
                        f"{dataset.dtype} -> {array.dtype}"
                    )

            old_count = self.frame_count
            new_count = old_count + 1
            try:
                for dataset in self._datasets.values():
                    dataset.resize((new_count,) + dataset.shape[1:])
                for path, array in leaves.items():
                    self._datasets[path][old_count] = array
            except Exception:
                for dataset in self._datasets.values():
                    if dataset.shape[0] == new_count:
                        dataset.resize((old_count,) + dataset.shape[1:])
                raise

            self.frame_count = new_count
            if new_count % self.flush_every_frames == 0:
                self._file.flush()

    def close(self):
        self.stop_recording()
