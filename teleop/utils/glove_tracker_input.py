"""Wuji Glove fingers and Vive Tracker wrist for the existing ZMQ teleop client."""

from __future__ import annotations

import time
from contextlib import suppress

import numpy as np

from utils.vive_tracker import ViveWristTracker
from utils.wuji_hand_spec import JOINT_LIMITS, validate_joint_vector


OPENVR_TO_ROBOT_TRACKING = np.array(
    [[-1.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def skeleton_keypoints(frame) -> np.ndarray:
    try:
        points = [
            [joint.pose.position.x, joint.pose.position.y, joint.pose.position.z]
            for joint in frame.joints
        ]
    except AttributeError:
        try:
            points = [list(joint.pose.position) for joint in frame.joints]
        except (AttributeError, TypeError) as exc:
            raise ValueError("Invalid Wuji Glove skeleton frame") from exc
    array = np.asarray(points, dtype=np.float32)
    if array.shape != (21, 3) or not np.isfinite(array).all():
        raise ValueError(f"Wuji Glove skeleton must be finite (21, 3), got {array.shape}")
    return array


def validated_glove_target(values) -> np.ndarray:
    target = validate_joint_vector(values, label="Wuji Glove retargeting")
    return np.clip(target, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


class GloveTrackerInput:
    def __init__(
        self,
        *,
        glove_serial: str,
        tracker_serial: str,
        user_id: str | None = None,
        tracker_to_wrist_xyz_m=(0.0, 0.0, 0.0),
        tracker_to_wrist_rpy_deg=(0.0, 0.0, 0.0),
        tracker_zero_frames: int = 20,
        startup_valid_frames: int = 10,
        tracking_timeout_s: float = 0.25,
    ):
        self.manager = None
        self.subscription = None
        self.tracker = None
        self.previous_user_id = None
        self.last_glove_time = None
        self.last_qpos = None
        self.tracking_timeout_s = float(tracking_timeout_s)
        if self.tracking_timeout_s <= 0:
            raise ValueError("Tracking timeout must be positive")
        try:
            from wuji_sdk import DeviceType, HandModel, Handedness, RetargetSession, SdkManager

            self.manager = SdkManager.instance()
            if user_id:
                self.previous_user_id = self.manager.current_user()["user_id"]
                self.manager.switch_user(user_id)
            gloves = [
                device for device in self.manager.scan()
                if device.device_type == DeviceType.WujiGlove
                and str(device.sn) == glove_serial
            ]
            if len(gloves) != 1:
                raise RuntimeError(f"Wuji Glove {glove_serial!r} was not found")
            glove = self.manager.connect(sn=gloves[0].sn, device_name="teleop_input_glove")
            if str(glove.hand_side().get()).split(".")[-1].lower() != "right":
                raise RuntimeError("This teleoperation path requires a right Wuji Glove")
            self.session = RetargetSession.for_hand(HandModel.WujiHand, side=Handedness.Right)
            self.subscription = glove.hand_skeleton().subscribe()
            self.tracker = ViveWristTracker(
                tracker_serial,
                tracker_to_wrist_xyz_m=tracker_to_wrist_xyz_m,
                tracker_to_wrist_rpy_deg=tracker_to_wrist_rpy_deg,
            )
            self.tracker.wait_and_zero(required_frames=tracker_zero_frames)
            self._wait_for_glove(startup_valid_frames)
        except BaseException:
            self.close()
            raise

    def _read_glove(self):
        frame = None
        while True:
            next_frame = self.subscription.recv()
            if next_frame is None:
                break
            frame = next_frame
        if frame is not None:
            self.last_qpos = validated_glove_target(
                self.session.step(skeleton_keypoints(frame))
            )
            self.last_glove_time = time.monotonic()
        return self.last_qpos

    def _wait_for_glove(self, required_frames: int):
        if required_frames < 1:
            raise ValueError("startup_valid_frames must be positive")
        valid_frames = 0
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            previous_time = self.last_glove_time
            try:
                self._read_glove()
                if self.last_glove_time != previous_time:
                    valid_frames += 1
            except (RuntimeError, ValueError):
                valid_frames = 0
            if valid_frames >= required_frames:
                return
            time.sleep(0.005)
        raise TimeoutError("Wuji Glove did not produce stable skeleton frames")

    def step(self) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            qpos = self._read_glove()
        except (RuntimeError, ValueError):
            return None
        world_wrist = self.tracker.read_world_wrist()
        now = time.monotonic()
        if (
            qpos is None
            or world_wrist is None
            or self.last_glove_time is None
            or now - self.last_glove_time > self.tracking_timeout_s
            or self.tracker.seconds_since_valid(now) > self.tracking_timeout_s
        ):
            return None
        wrist = OPENVR_TO_ROBOT_TRACKING @ world_wrist
        if not np.isfinite(wrist).all():
            return None
        return wrist, qpos.copy()

    def close(self):
        if self.tracker is not None:
            with suppress(Exception):
                self.tracker.close()
            self.tracker = None
        if self.subscription is not None:
            with suppress(Exception):
                self.subscription.close()
            self.subscription = None
        if self.manager is not None:
            with suppress(Exception):
                self.manager.disconnect_all()
            if self.previous_user_id is not None:
                with suppress(Exception):
                    self.manager.switch_user(self.previous_user_id)
            self.manager = None
