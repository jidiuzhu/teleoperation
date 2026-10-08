#!/usr/bin/env python3
"""Deploy the multimodal ACTBH policy to RM75 + Wuji with safe shadow defaults."""

from __future__ import annotations

import os

for thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_env_var, "1")

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch


TELEOP_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = TELEOP_ROOT.parent
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

from utils.actbh_observation import (
    ACTBHObservationConfig,
    ACTION_KEY,
    DEPTH_IMAGE_KEY,
    ENVIRONMENT_STATE_KEY,
    POINT_CLOUD_KEY,
    RGB_KEY,
    STATE_KEY,
    LiveACTBHObservationBuilder,
    assemble_actbh_observation,
)
from utils.kinect_hand_eye import load_T_base_camera
from utils.wuji_hand_spec import JOINT_LIMITS as WUJI_URDF_LIMITS


DEFAULT_CHECKPOINT = WORKSPACE_ROOT / "outputs/train/latest/pretrained_model"
DEFAULT_HAND_EYE = TELEOP_ROOT / "kinect_hand_eye.json"
DEFAULT_ARM_URDF = (
    TELEOP_ROOT / "assets/RM75/urdf/RM75-6F/urdf/RM75-6F.urdf"
)
MOTION_CONFIRMATION = "I_UNDERSTAND_THIS_MOVES_THE_ROBOT"
EXPECTED_INPUT_SHAPES = {
    STATE_KEY: (27,),
    ENVIRONMENT_STATE_KEY: (15,),
    RGB_KEY: (3, 240, 320),
    DEPTH_IMAGE_KEY: (3, 240, 320),
    POINT_CLOUD_KEY: (1024, 3),
}


class SafetyStop(RuntimeError):
    """Raised when a command must not be sent to the real robot."""


@dataclass(frozen=True)
class MotionSafetyConfig:
    arm_limits_rad: np.ndarray
    hand_limits_rad: np.ndarray
    arm_max_step_rad: float
    hand_max_step_rad: float
    arm_tracking_error_rad: float
    arm_alpha: float
    hand_alpha: float

    def validate(self) -> None:
        for name, limits, shape in (
            ("arm", self.arm_limits_rad, (7, 2)),
            ("hand", self.hand_limits_rad, (20, 2)),
        ):
            values = np.asarray(limits, dtype=np.float64)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"{name} limits must have finite shape {shape}")
            if np.any(values[:, 0] >= values[:, 1]):
                raise ValueError(f"{name} limits contain lower >= upper")
        if self.arm_max_step_rad <= 0 or self.hand_max_step_rad <= 0:
            raise ValueError("per-step motion limits must be positive")
        if self.arm_tracking_error_rad <= 0:
            raise ValueError("arm tracking error limit must be positive")
        if not 0 < self.arm_alpha <= 1 or not 0 < self.hand_alpha <= 1:
            raise ValueError("smoothing alpha values must be in (0, 1]")


class MotionSafetyFilter:
    """Validate, limit, and smooth absolute joint targets before hardware I/O."""

    def __init__(self, config: MotionSafetyConfig):
        config.validate()
        self.config = config
        self.last_arm_command = None
        self.last_hand_command = None

    @staticmethod
    def _joint_vector(value, size: int, label: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,) or not np.isfinite(vector).all():
            raise SafetyStop(f"{label} must have finite shape ({size},), got {vector.shape}")
        return vector

    def reset(self, arm_q_rad: np.ndarray, hand_q_rad: np.ndarray) -> None:
        self.last_arm_command = self._joint_vector(arm_q_rad, 7, "arm feedback").copy()
        self.last_hand_command = self._joint_vector(hand_q_rad, 20, "hand feedback").copy()

    def filter(
        self,
        action: np.ndarray,
        arm_q_rad: np.ndarray,
        hand_q_rad: np.ndarray,
        *,
        enforce_tracking: bool,
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        command = self._joint_vector(action, 27, "ACTBH action")
        arm_q = self._joint_vector(arm_q_rad, 7, "arm feedback")
        hand_q = self._joint_vector(hand_q_rad, 20, "hand feedback")
        if self.last_arm_command is None or self.last_hand_command is None:
            self.reset(arm_q, hand_q)

        if enforce_tracking:
            tracking_error = float(np.max(np.abs(arm_q - self.last_arm_command)))
            if tracking_error > self.config.arm_tracking_error_rad:
                raise SafetyStop(
                    "RM75 tracking error exceeded limit: "
                    f"{np.rad2deg(tracking_error):.2f} deg > "
                    f"{np.rad2deg(self.config.arm_tracking_error_rad):.2f} deg"
                )
            arm_base = self.last_arm_command
            hand_base = self.last_hand_command
        else:
            tracking_error = 0.0
            arm_base = arm_q
            hand_base = hand_q

        raw_arm = command[:7]
        raw_hand = command[7:]
        arm_clipped = np.clip(
            raw_arm,
            self.config.arm_limits_rad[:, 0],
            self.config.arm_limits_rad[:, 1],
        )
        hand_clipped = np.clip(
            raw_hand,
            self.config.hand_limits_rad[:, 0],
            self.config.hand_limits_rad[:, 1],
        )
        arm_smoothed = arm_base + self.config.arm_alpha * (arm_clipped - arm_base)
        hand_smoothed = hand_base + self.config.hand_alpha * (hand_clipped - hand_base)
        safe_arm = arm_base + np.clip(
            arm_smoothed - arm_base,
            -self.config.arm_max_step_rad,
            self.config.arm_max_step_rad,
        )
        safe_hand = hand_base + np.clip(
            hand_smoothed - hand_base,
            -self.config.hand_max_step_rad,
            self.config.hand_max_step_rad,
        )
        safe_arm = np.clip(
            safe_arm,
            self.config.arm_limits_rad[:, 0],
            self.config.arm_limits_rad[:, 1],
        )
        safe_hand = np.clip(
            safe_hand,
            self.config.hand_limits_rad[:, 0],
            self.config.hand_limits_rad[:, 1],
        )
        if enforce_tracking:
            self.last_arm_command = safe_arm.copy()
            self.last_hand_command = safe_hand.copy()

        safe_action = np.concatenate([safe_arm, safe_hand]).astype(np.float32)
        diagnostics = {
            "arm_limit_clipped": bool(not np.allclose(raw_arm, arm_clipped)),
            "hand_limit_clipped": bool(not np.allclose(raw_hand, hand_clipped)),
            "arm_max_step_rad": float(np.max(np.abs(safe_arm - arm_base))),
            "hand_max_step_rad": float(np.max(np.abs(safe_hand - hand_base))),
            "arm_tracking_error_rad": tracking_error,
        }
        return safe_action, diagnostics


class JsonlLogger:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def load_arm_limits(urdf_path: Path, margin_rad: float) -> np.ndarray:
    root = ET.parse(urdf_path).getroot()
    joints = {joint.attrib.get("name"): joint for joint in root.findall("joint")}
    limits = []
    for index in range(1, 8):
        name = f"joint_{index}"
        joint = joints.get(name)
        limit = None if joint is None else joint.find("limit")
        if joint is None or joint.attrib.get("type") != "revolute" or limit is None:
            raise ValueError(f"Missing revolute RM75 limit for {name} in {urdf_path}")
        limits.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))
    result = np.asarray(limits, dtype=np.float64)
    result[:, 0] += margin_rad
    result[:, 1] -= margin_rad
    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError("RM75 joint margin leaves an empty safe range")
    return result


def resolve_device(requested: str) -> torch.device:
    value = requested.lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda was requested but torch.cuda.is_available() is false")
    if value not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return torch.device(value)


def load_policy_runtime(checkpoint: Path, device: torch.device, n_action_steps: int):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.actbh.configuration_actbh import ACTBHConfig
    from lerobot.policies.actbh.modeling_actbh import ACTBHPolicy
    from lerobot.policies.factory import make_pre_post_processors

    model_dir = checkpoint.expanduser().resolve()
    required = (
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint {model_dir} is missing: {missing}")

    config = PreTrainedConfig.from_pretrained(model_dir, local_files_only=True)
    if not isinstance(config, ACTBHConfig):
        raise TypeError(f"Expected an ACTBH checkpoint, loaded {type(config).__name__}")
    actual_shapes = {key: tuple(feature.shape) for key, feature in config.input_features.items()}
    if actual_shapes != EXPECTED_INPUT_SHAPES:
        raise ValueError(
            f"Checkpoint input contract mismatch: expected {EXPECTED_INPUT_SHAPES}, got {actual_shapes}"
        )
    if config.chunk_size < n_action_steps:
        raise ValueError(
            f"n_action_steps={n_action_steps} exceeds checkpoint chunk_size={config.chunk_size}"
        )
    config.device = str(device)
    config.n_action_steps = int(n_action_steps)
    policy = ACTBHPolicy.from_pretrained(model_dir, config=config, local_files_only=True)
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(model_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, preprocessor, postprocessor, config


def infer_action(observation, policy, preprocessor, postprocessor, device, use_amp):
    from lerobot.utils.control_utils import predict_action

    started = time.perf_counter()
    action = predict_action(
        observation=observation,
        policy=policy,
        device=device,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        use_amp=use_amp,
        task="Grasp the apple",
        robot_type="rm75_wuji",
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if result.shape != (27,) or not np.isfinite(result).all():
        raise SafetyStop(f"Model returned invalid action shape/value: {result.shape}")
    return result, elapsed_ms


def make_safety_filter(args, hand_limits: np.ndarray) -> MotionSafetyFilter:
    arm_limits = load_arm_limits(args.arm_urdf, np.deg2rad(args.arm_limit_margin_deg))
    hand_limits = np.asarray(hand_limits, dtype=np.float64).copy()
    hand_limits[:, 0] += args.hand_limit_margin_rad
    hand_limits[:, 1] -= args.hand_limit_margin_rad
    return MotionSafetyFilter(
        MotionSafetyConfig(
            arm_limits_rad=arm_limits,
            hand_limits_rad=hand_limits,
            arm_max_step_rad=np.deg2rad(args.arm_max_step_deg),
            hand_max_step_rad=args.hand_max_step_rad,
            arm_tracking_error_rad=np.deg2rad(args.arm_tracking_error_deg),
            arm_alpha=args.arm_alpha,
            hand_alpha=args.hand_alpha,
        )
    )


def replay_h5(args, runtime, device: torch.device, logger: JsonlLogger) -> None:
    policy, preprocessor, postprocessor, config = runtime
    source = args.replay_h5.expanduser().resolve()
    observation_config = ACTBHObservationConfig()
    errors = []
    with h5py.File(source, "r") as h5_file:
        if "kinect_T_base_camera" not in h5_file.attrs:
            raise KeyError(f"Replay file has no kinect_T_base_camera: {source}")
        base_camera = np.asarray(h5_file.attrs["kinect_T_base_camera"])
        total_frames = len(h5_file["time"])
        stop = total_frames if args.replay_frames == 0 else min(
            total_frames, args.replay_start + args.replay_frames
        )
        if not 0 <= args.replay_start < stop:
            raise ValueError(
                f"Invalid replay range start={args.replay_start}, stop={stop}, frames={total_frames}"
            )
        policy.reset()
        for frame_index in range(args.replay_start, stop):
            if not bool(h5_file["observations/kinect/valid"][frame_index]):
                raise ValueError(f"Replay frame {frame_index} has invalid Kinect data")
            if "observations/kinect/segmentation_valid" in h5_file and not bool(
                h5_file["observations/kinect/segmentation_valid"][frame_index]
            ):
                raise ValueError(f"Replay frame {frame_index} has invalid segmentation data")
            qpos = np.asarray(h5_file["observations/qpos"][frame_index], dtype=np.float32)
            point_count = int(h5_file["observations/kinect/pointcloud_count"][frame_index])
            observation = assemble_actbh_observation(
                rgb_hwc=h5_file["observations/images/KINECT/rgb"][frame_index],
                depth_mm=h5_file["observations/images/KINECT/depth"][frame_index],
                arm_q_rad=qpos[:7],
                hand_q_rad=qpos[7:],
                distances_m=h5_file["observations/distances"][frame_index],
                pointcloud_camera=h5_file["observations/kinect/pointcloud_xyz"][frame_index],
                pointcloud_count=point_count,
                T_base_camera=base_camera,
                config=observation_config,
            )
            action, inference_ms = infer_action(
                observation,
                policy,
                preprocessor,
                postprocessor,
                device,
                config.use_amp,
            )
            reference = np.concatenate(
                [
                    h5_file["action/arm_target_q"][frame_index],
                    h5_file["action/hand_command_q"][frame_index],
                ]
            ).astype(np.float32)
            absolute_error = np.abs(action - reference)
            errors.append(absolute_error)
            logger.write(
                {
                    "mode": "replay",
                    "source": str(source),
                    "frame_index": frame_index,
                    "inference_ms": inference_ms,
                    "predicted_action": action.tolist(),
                    "reference_action": reference.tolist(),
                    "mae": float(absolute_error.mean()),
                }
            )
            print(
                f"[Replay] frame={frame_index} inference={inference_ms:.1f}ms "
                f"arm_mae={absolute_error[:7].mean():.4f}rad "
                f"hand_mae={absolute_error[7:].mean():.4f}rad",
                flush=True,
            )
    stacked = np.stack(errors)
    print(
        f"[Replay] complete frames={len(stacked)} overall_mae={stacked.mean():.5f}rad "
        f"arm_mae={stacked[:, :7].mean():.5f}rad "
        f"hand_mae={stacked[:, 7:].mean():.5f}rad",
        flush=True,
    )


def wait_for_kinect(receiver, timeout_s: float) -> dict:
    started = time.monotonic()
    while time.monotonic() - started < timeout_s:
        frame = receiver.get_frames().get("KINECT")
        if frame is not None and frame.get("valid", False):
            return frame
        time.sleep(0.05)
    raise TimeoutError("No valid Kinect segmentation sample; start the publisher first")


def check_live_dependencies(hand_backend: str) -> None:
    missing = []
    if importlib.util.find_spec("Robotic_Arm") is None:
        missing.append("robotic-arm==1.1.6 (provides Robotic_Arm)")
    if hand_backend in {"wujihandpy", ""} and importlib.util.find_spec("wujihandpy") is None:
        missing.append("wujihandpy==1.8.0")
    if missing:
        raise RuntimeError(
            "Live deployment dependencies are missing from this Python environment: "
            + ", ".join(missing)
            + ". Install them into the same environment that runs LeRobot; do not "
            "reuse a different Python version's site-packages directory."
        )


def run_live(args, runtime, device: torch.device, logger: JsonlLogger) -> None:
    check_live_dependencies(args.hand_backend)
    from utils.Realman_controller import RoboticArmController
    from utils.XHand_controller_wuji import WujiHandDriver
    from utils.kinect_stream import KinectZmqReceiver
    from utils.wuji_fingertip_kinematics import WujiFingertipKinematics
    from utils.wuji_keypoint_visualization import DEFAULT_KEYPOINT_SPECS

    policy, preprocessor, postprocessor, config = runtime
    arm = None
    hand = None
    receiver = None
    stopping = False

    def request_stop(_signal, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        print(f"[Live] Connecting RM75 at {args.arm_ip}:{args.arm_port}", flush=True)
        arm = RoboticArmController(ip=args.arm_ip, port=args.arm_port)
        print(f"[Live] Connecting Wuji backend={args.hand_backend}", flush=True)
        hand = WujiHandDriver(device=args.hand_device, backend_spec=args.hand_backend)
        hardware_hand_limits = np.asarray(hand.get_joint_limits(), dtype=np.float64)
        combined_hand_limits = np.column_stack(
            [
                np.maximum(WUJI_URDF_LIMITS[:, 0], hardware_hand_limits[:, 0]),
                np.minimum(WUJI_URDF_LIMITS[:, 1], hardware_hand_limits[:, 1]),
            ]
        )
        safety_filter = make_safety_filter(args, combined_hand_limits)
        receiver = KinectZmqReceiver(
            args.kinect_addr,
            max_age_s=args.kinect_max_age,
            pointcloud_capacity=args.pointcloud_capacity,
        )
        wait_for_kinect(receiver, args.startup_timeout)
        builder = LiveACTBHObservationBuilder(
            T_base_camera=load_T_base_camera(args.kinect_hand_eye),
            keypoint_kinematics=WujiFingertipKinematics(),
            keypoint_specs=DEFAULT_KEYPOINT_SPECS,
            flange_thickness_m=args.flange_thickness,
        )
        arm_ok, arm_deg = arm.get_joint_state()
        if not arm_ok:
            raise RuntimeError("Cannot read initial RM75 state")
        arm_q = np.deg2rad(np.asarray(arm_deg, dtype=np.float64))
        hand_q = np.asarray(hand.get_current_joints(), dtype=np.float64)
        safety_filter.reset(arm_q, hand_q)
        mode = "MOTION ENABLED" if args.enable_motion else "SHADOW (no commands sent)"
        print(f"[Live] {mode}; control={args.control_hz:.1f}Hz; Ctrl+C stops", flush=True)
        if args.enable_motion and args.motion_start_delay > 0:
            print(
                f"[Live] Motion begins in {args.motion_start_delay:.1f}s; "
                "verify the workspace and physical emergency stop now",
                flush=True,
            )
            time.sleep(args.motion_start_delay)
        policy.reset()
        next_deadline = time.perf_counter()
        step = 0
        invalid_latched = False
        while not stopping and (args.max_steps == 0 or step < args.max_steps):
            loop_started = time.perf_counter()
            arm_ok, arm_deg = arm.get_joint_state()
            if not arm_ok:
                raise SafetyStop("RM75 feedback read failed")
            arm_q = np.deg2rad(np.asarray(arm_deg, dtype=np.float64))
            hand_q = np.asarray(hand.get_current_joints(), dtype=np.float64)
            frame = receiver.get_frames().get("KINECT")
            try:
                observation = builder.build(arm_q, hand_q, frame)
                invalid_latched = False
            except (KeyError, TypeError, ValueError) as exc:
                if not invalid_latched:
                    print(f"[Live] Holding: {exc}", flush=True)
                    policy.reset()
                    invalid_latched = True
                time.sleep(min(0.05, 1.0 / args.control_hz))
                continue

            action, inference_ms = infer_action(
                observation,
                policy,
                preprocessor,
                postprocessor,
                device,
                config.use_amp,
            )
            safe_action, safety_diag = safety_filter.filter(
                action,
                arm_q,
                hand_q,
                enforce_tracking=args.enable_motion,
            )
            if args.enable_motion:
                arm.move_j_stream(np.rad2deg(safe_action[:7]))
                hand.send_joints(safe_action[7:])
            logger.write(
                {
                    "mode": "live_motion" if args.enable_motion else "live_shadow",
                    "step": step,
                    "frame_id": int(frame.get("frame_id", -1)),
                    "sync_age_ms": float(frame.get("sync_age_ms", -1.0)),
                    "inference_ms": inference_ms,
                    "arm_feedback_rad": arm_q.tolist(),
                    "hand_feedback_rad": hand_q.tolist(),
                    "predicted_action": action.tolist(),
                    "safe_action": safe_action.tolist(),
                    "safety": safety_diag,
                }
            )
            if step % max(1, int(args.control_hz)) == 0:
                print(
                    f"[Live] step={step} frame={frame.get('frame_id')} "
                    f"age={frame.get('sync_age_ms', -1):.1f}ms "
                    f"inference={inference_ms:.1f}ms mode={mode}",
                    flush=True,
                )
            step += 1
            next_deadline += 1.0 / args.control_hz
            sleep_s = next_deadline - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            elif time.perf_counter() - loop_started > 1.0 / args.control_hz:
                next_deadline = time.perf_counter()
    finally:
        policy.reset()
        if receiver is not None:
            receiver.stop()
        if arm is not None:
            arm.disconnect()
        if hand is not None:
            hand.close()
        print("[Live] Hardware connections closed", flush=True)


def default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return WORKSPACE_ROOT / "outputs/deploy" / f"actbh_{timestamp}.jsonl"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ACTBH deployment for RM75 + Wuji; shadow mode is the default"
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--n-action-steps", type=int, default=3)
    parser.add_argument("--log-jsonl", type=Path, default=None)
    parser.add_argument("--replay-h5", type=Path)
    parser.add_argument("--replay-start", type=int, default=0)
    parser.add_argument("--replay-frames", type=int, default=10, help="0 replays all remaining frames")
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument("--motion-confirmation", default="")
    parser.add_argument("--motion-start-delay", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--control-hz", type=float, default=15.0)
    parser.add_argument("--arm-ip", default="192.0.2.1")
    parser.add_argument("--arm-port", type=int, default=8080)
    parser.add_argument("--arm-urdf", type=Path, default=DEFAULT_ARM_URDF)
    parser.add_argument("--hand-backend", default="wujihandpy")
    parser.add_argument("--hand-device", default=None)
    parser.add_argument("--kinect-addr", default="tcp://127.0.0.1:6200")
    parser.add_argument("--kinect-hand-eye", type=Path, default=DEFAULT_HAND_EYE)
    parser.add_argument("--kinect-max-age", type=float, default=0.2)
    parser.add_argument("--startup-timeout", type=float, default=30.0)
    parser.add_argument("--pointcloud-capacity", type=int, default=8192)
    parser.add_argument("--flange-thickness", type=float, default=0.036)
    parser.add_argument("--arm-limit-margin-deg", type=float, default=1.0)
    parser.add_argument("--hand-limit-margin-rad", type=float, default=0.01)
    parser.add_argument("--arm-max-step-deg", type=float, default=2.0)
    parser.add_argument("--hand-max-step-rad", type=float, default=0.1)
    parser.add_argument("--arm-tracking-error-deg", type=float, default=10.0)
    parser.add_argument("--arm-alpha", type=float, default=0.3)
    parser.add_argument("--hand-alpha", type=float, default=0.7)
    args = parser.parse_args(argv)

    positive = {
        "n_action_steps": args.n_action_steps,
        "control_hz": args.control_hz,
        "kinect_max_age": args.kinect_max_age,
        "startup_timeout": args.startup_timeout,
        "pointcloud_capacity": args.pointcloud_capacity,
        "arm_max_step_deg": args.arm_max_step_deg,
        "hand_max_step_rad": args.hand_max_step_rad,
        "arm_tracking_error_deg": args.arm_tracking_error_deg,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if (
        args.replay_start < 0
        or args.replay_frames < 0
        or args.max_steps < 0
        or args.motion_start_delay < 0
    ):
        parser.error("replay counts, max steps, and motion start delay must be non-negative")
    if args.arm_limit_margin_deg < 0 or args.hand_limit_margin_rad < 0:
        parser.error("joint-limit margins must be non-negative")
    if args.flange_thickness < 0:
        parser.error("--flange-thickness must be non-negative")
    if args.enable_motion and args.replay_h5:
        parser.error("--enable-motion cannot be combined with --replay-h5")
    if args.enable_motion and args.motion_confirmation != MOTION_CONFIRMATION:
        parser.error(
            "--enable-motion requires --motion-confirmation=" + MOTION_CONFIRMATION
        )
    if not 0 < args.arm_alpha <= 1 or not 0 < args.hand_alpha <= 1:
        parser.error("--arm-alpha and --hand-alpha must be in (0, 1]")
    return args


def main(argv=None):
    args = parse_args(argv)
    log_path = (args.log_jsonl or default_log_path()).expanduser().resolve()
    try:
        if args.replay_h5 is None:
            check_live_dependencies(args.hand_backend)
        device = resolve_device(args.device)
        print(f"[ACTBH] checkpoint={args.checkpoint.expanduser().resolve()}", flush=True)
        print(f"[ACTBH] device={device} n_action_steps={args.n_action_steps}", flush=True)
        print(f"[ACTBH] log={log_path}", flush=True)
        runtime = load_policy_runtime(args.checkpoint, device, args.n_action_steps)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"[STARTUP ERROR] {exc}", file=sys.stderr, flush=True)
        return 1
    logger = JsonlLogger(log_path)
    try:
        if args.replay_h5:
            replay_h5(args, runtime, device, logger)
        else:
            run_live(args, runtime, device, logger)
    except SafetyStop as exc:
        print(f"[SAFETY STOP] {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
