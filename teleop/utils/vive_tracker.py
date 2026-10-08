"""OpenVR-based Vive Tracker wrist pose acquisition."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class RelativeWristPose:
    transform: np.ndarray
    xyz_m: np.ndarray
    rpy_deg: np.ndarray


def matrix34_to_transform(matrix: Any) -> np.ndarray:
    array = np.asarray([[matrix[row][col] for col in range(4)] for row in range(3)])
    if array.shape != (3, 4) or not np.all(np.isfinite(array)):
        raise ValueError("OpenVR pose matrix must be finite and have shape (3, 4)")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :] = array
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise ValueError("OpenVR pose contains an invalid rotation matrix")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise ValueError("OpenVR pose rotation must be right-handed")
    return transform


def make_transform(xyz_m: Any, rpy_deg: Any) -> np.ndarray:
    xyz = np.asarray(xyz_m, dtype=np.float64)
    rpy = np.asarray(rpy_deg, dtype=np.float64)
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError("xyz and roll/pitch/yaw must each contain three values")
    if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(rpy)):
        raise ValueError("tracker-to-wrist transform contains NaN or infinity")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
    transform[:3, 3] = xyz
    return transform


def average_transforms(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("at least one transform is required")
    stack = np.asarray(transforms, dtype=np.float64)
    if stack.ndim != 3 or stack.shape[1:] != (4, 4):
        raise ValueError("transforms must have shape (N, 4, 4)")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean(stack[:, :3, 3], axis=0)
    result[:3, :3] = Rotation.from_matrix(stack[:, :3, :3]).mean().as_matrix()
    return result


def relative_wrist_pose(initial_world_wrist: Any, current_world_wrist: Any) -> RelativeWristPose:
    initial = np.asarray(initial_world_wrist, dtype=np.float64)
    current = np.asarray(current_world_wrist, dtype=np.float64)
    if initial.shape != (4, 4) or current.shape != (4, 4):
        raise ValueError("wrist transforms must have shape (4, 4)")
    relative = np.linalg.inv(initial) @ current
    xyz = relative[:3, 3].copy()
    rpy = Rotation.from_matrix(relative[:3, :3]).as_euler("xyz", degrees=True)
    return RelativeWristPose(relative, xyz, rpy)


def format_relative_wrist(pose: RelativeWristPose) -> str:
    xyz_mm = pose.xyz_m * 1000.0
    roll, pitch, yaw = pose.rpy_deg
    return (
        "wrist_rel "
        f"xyz_mm=[{xyz_mm[0]:+7.1f}, {xyz_mm[1]:+7.1f}, {xyz_mm[2]:+7.1f}] "
        f"rpy_deg=[{roll:+7.1f}, {pitch:+7.1f}, {yaw:+7.1f}]"
    )


class ViveWristTracker:
    """Read one GenericTracker and express wrist motion relative to startup."""

    def __init__(
        self,
        serial: str,
        *,
        tracker_to_wrist_xyz_m: Any = (0.0, 0.0, 0.0),
        tracker_to_wrist_rpy_deg: Any = (0.0, 0.0, 0.0),
    ) -> None:
        try:
            import openvr
        except ImportError as exc:
            raise RuntimeError(
                "openvr is not installed; run: python -m pip install -r requirements.txt"
            ) from exc

        self.openvr = openvr
        try:
            self.system = openvr.init(openvr.VRApplication_Other)
        except Exception as exc:
            raise RuntimeError(
                "Could not connect to SteamVR; start SteamVR before teleoperation"
            ) from exc
        self.serial = serial
        self.tracker_to_wrist = make_transform(
            tracker_to_wrist_xyz_m, tracker_to_wrist_rpy_deg
        )
        self.device_index: int | None = None
        self.ever_connected = False
        self.ever_valid = False
        self.initial_world_wrist: np.ndarray | None = None
        self.last_relative: RelativeWristPose | None = None
        self.last_valid_time: float | None = None
        self.closed = False

    @staticmethod
    def _text(value: Any) -> str:
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    def _find_device(self) -> int | None:
        for index in range(self.openvr.k_unMaxTrackedDeviceCount):
            if not self.system.isTrackedDeviceConnected(index):
                continue
            if self.system.getTrackedDeviceClass(index) != self.openvr.TrackedDeviceClass_GenericTracker:
                continue
            serial = self._text(
                self.system.getStringTrackedDeviceProperty(
                    index, self.openvr.Prop_SerialNumber_String
                )
            )
            if serial == self.serial:
                self.device_index = index
                self.ever_connected = True
                return index
        self.device_index = None
        return None

    def read_world_wrist(self) -> np.ndarray | None:
        index = self.device_index
        if index is None or not self.system.isTrackedDeviceConnected(index):
            index = self._find_device()
        if index is None:
            return None

        poses = self.system.getDeviceToAbsoluteTrackingPose(
            self.openvr.TrackingUniverseStanding,
            0.0,
            self.openvr.k_unMaxTrackedDeviceCount,
        )
        pose = poses[index]
        if not pose.bDeviceIsConnected:
            self.device_index = None
            return None
        if not pose.bPoseIsValid:
            return None
        if pose.eTrackingResult != self.openvr.TrackingResult_Running_OK:
            return None

        world_tracker = matrix34_to_transform(pose.mDeviceToAbsoluteTracking)
        self.ever_valid = True
        self.last_valid_time = time.monotonic()
        return world_tracker @ self.tracker_to_wrist

    def wait_and_zero(
        self,
        *,
        required_frames: int = 20,
        timeout_seconds: float = 10.0,
        max_position_spread_m: float = 0.02,
        max_rotation_spread_deg: float = 5.0,
    ) -> None:
        samples: deque[np.ndarray] = deque(maxlen=required_frames)
        deadline = time.monotonic() + timeout_seconds
        print(
            f"Waiting for {required_frames} stable Vive Tracker frames "
            f"from {self.serial}..."
        )
        while time.monotonic() < deadline:
            transform = self.read_world_wrist()
            if transform is None:
                samples.clear()
                time.sleep(0.01)
                continue
            samples.append(transform)
            if len(samples) < required_frames:
                time.sleep(0.005)
                continue

            sample_list = list(samples)
            mean = average_transforms(sample_list)
            positions = np.asarray([item[:3, 3] for item in sample_list])
            max_position_spread = float(
                np.max(np.linalg.norm(positions - mean[:3, 3], axis=1))
            )
            mean_rotation = Rotation.from_matrix(mean[:3, :3])
            sample_rotations = Rotation.from_matrix(
                np.asarray([item[:3, :3] for item in sample_list])
            )
            max_rotation_spread = float(
                np.max((mean_rotation.inv() * sample_rotations).magnitude())
            )
            if (
                max_position_spread <= max_position_spread_m
                and max_rotation_spread <= np.deg2rad(max_rotation_spread_deg)
            ):
                self.initial_world_wrist = mean
                self.last_relative = relative_wrist_pose(mean, mean)
                print(
                    "Vive Tracker zeroed: keep the glove still until Wuji Hand "
                    "startup completes."
                )
                return
            samples.clear()
        if not self.ever_connected:
            detail = "was not connected to SteamVR"
        elif not self.ever_valid:
            detail = "was connected but never obtained a valid Lighthouse pose"
        else:
            detail = "did not remain still long enough to establish the zero pose"
        raise TimeoutError(f"Vive Tracker {self.serial} {detail}")

    def read_relative(self) -> RelativeWristPose | None:
        if self.initial_world_wrist is None:
            raise RuntimeError("Vive Tracker must be zeroed before reading relative motion")
        current = self.read_world_wrist()
        if current is None:
            return None
        self.last_relative = relative_wrist_pose(self.initial_world_wrist, current)
        return self.last_relative

    def seconds_since_valid(self, now: float | None = None) -> float:
        if self.last_valid_time is None:
            return float("inf")
        timestamp = time.monotonic() if now is None else now
        return timestamp - self.last_valid_time

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.openvr.shutdown()
