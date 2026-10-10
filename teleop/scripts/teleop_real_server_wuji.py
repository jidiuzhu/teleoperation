#!/usr/bin/env python3
"""
Teleoperation Server for RM75 / MARVIN + Wuji Hand
Responsibility:
1. Hardware Interface (Arm, Hand, Camera).
2. Kinematics Solver (IK/FK).
3. Data Collection (HDF5).
4. ZMQ Communication with Client.
"""

import os

# This 7-DoF IK problem is too small to benefit from threaded BLAS. Allowing
# each native library to create a full CPU-sized pool causes severe
# oversubscription and makes IPOPT/MUMPS much slower. These variables must be
# set before importing NumPy, SciPy, Pinocchio, CasADi, or OpenCV.
for thread_env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[thread_env_var] = "1"

import argparse
import time
import signal
import sys
import json
from pathlib import Path
import numpy as np
import zmq
import cv2
import threading
from queue import Empty, Queue
from scipy.spatial.transform import Rotation as R

# OpenCV has an independent worker pool that is not consistently controlled
# by the BLAS environment variables above.
cv2.setNumThreads(1)

TELEOP_ROOT = Path(__file__).resolve().parents[1]
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

# --- Import Utilities ---
try:
    from utils.XHand_controller_wuji import (
        MAX_JOINT_EFFORT_LIMIT_A,
        WujiHandDriver,
    )
    from utils.data_collector_wuji import DataCollectorWuji
    from utils.wuji_fingertip_kinematics import WujiFingertipKinematics
    from utils.wuji_keypoint_visualization import DEFAULT_KEYPOINT_SPECS
    from utils.wuji_hand_spec import (
        hand_spec,
        HAND_DOF,
        HAND_MODEL,
        HOME_Q,
        JOINT_LIMITS,
        JOINT_NAMES,
        URDF_PATH as WUJI_URDF_PATH,
        validate_joint_vector,
    )
except ImportError as e:
    print(f"[Fatal] Dependency Import Failed: {e}")
    print("Please install the existing arm, camera, and kinematics runtime dependencies.")
    sys.exit(1)

# ================= Configuration =================

# --- 1. Robot Arm Config ---
ROBOT_IP = "192.0.2.1"
ROBOT_PORT = 8080
URDF_PATH = str(
    TELEOP_ROOT / "assets" / "RM75" / "urdf" / "RM75-6F" / "urdf" / "RM75-6F.urdf"
)

# Safety & Control Parameters
ARM_MOTION_SCALE = 1.0  # Scaling factor for human hand motion
Z_MIN_HEIGHT = 0.05  # Virtual table height limit (meters)
ARM_SMOOTH_ALPHA = 0.3  # Low-pass filter for arm joints (0.0=No change, 1.0=Instant)
MAX_JOINT_ERROR_DEG = 60.0  # Safety fuse: stop if real robot lags too far behind target
STARTUP_GRACE_PERIOD = 3.0  # Time (s) to ignore large errors during startup
MAX_EE_DISPLACEMENT_M = 0.45
MAX_EE_LINEAR_SPEED_M_S = 0.90
MAX_EE_ANGULAR_SPEED_DEG_S = 225.0
COMMAND_WATCHDOG_S = 0.5

# Rotation Correction (if client/retargeting is aligned, set to 0)
FIXED_OFFSET_DEG = 0.0

# --- 2. Dexterous Hand Config ---
HAND_SMOOTH_ALPHA = 0.3

# The final limits are the intersection of the conservative URDF limits and
# the per-device calibrated limits reported by wujihandpy.
HAND_LIMIT_MARGIN = 0.01

# Measure the installed adapter and override with --flange-thickness if needed.
DEFAULT_FLANGE_THICKNESS = 0.036

# --- 3. System Config ---
CONTROL_HZ = 15.0  # Main control loop frequency
IMG_HZ = 10.0  # Image streaming frequency
DEFAULT_KINECT_HAND_EYE_PATH = TELEOP_ROOT / "kinect_hand_eye.json"


def limit_pose_step(current_pose, desired_pose, max_translation, max_rotation_rad):
    """Limit one Cartesian target step while preserving a valid quaternion."""
    current = np.asarray(current_pose, dtype=np.float64)
    desired = np.asarray(desired_pose, dtype=np.float64)
    if current.shape != (7,) or desired.shape != (7,):
        raise ValueError("current_pose and desired_pose must have shape (7,)")

    translation = desired[:3] - current[:3]
    translation_norm = np.linalg.norm(translation)
    if translation_norm > max_translation:
        translation *= max_translation / translation_norm
    next_position = current[:3] + translation

    current_rotation = R.from_quat(current[3:])
    desired_rotation = R.from_quat(desired[3:])
    rotation_step = (desired_rotation * current_rotation.inv()).as_rotvec()
    rotation_angle = np.linalg.norm(rotation_step)
    if rotation_angle > max_rotation_rad:
        rotation_step *= max_rotation_rad / rotation_angle
    next_rotation = R.from_rotvec(rotation_step) * current_rotation
    return np.concatenate([next_position, next_rotation.as_quat()])


class CombinedCameraReader:
    def __init__(self, rs_reader, zed_reader, kinect_reader=None):
        self.rs = rs_reader
        self.zed = zed_reader
        self.kinect = kinect_reader

    def get_frames(self):
        frames = {}
        if self.rs:
            try:
                rs_frames = self.rs.get_frames()
                if rs_frames:
                    frames.update(rs_frames)
            except Exception as e:
                pass

        if self.zed:
            try:
                zed_frames = self.zed.get_frames()
                if zed_frames:
                    frames.update(zed_frames)
            except Exception as e:
                pass

        if self.kinect:
            try:
                kinect_frames = self.kinect.get_frames()
                if kinect_frames:
                    frames.update(kinect_frames)
            except Exception:
                pass

        return frames

    def stop(self):
        if self.rs: self.rs.stop()
        if self.zed: self.zed.stop()
        if self.kinect: self.kinect.stop()


class AsyncHardwareInterface:
    def __init__(self, arm, hand, camera):
        self.arm = arm
        self.hand = hand
        self.camera = camera

        self.running = True
        self.lock = threading.Lock()
        self.diagnostics_lock = threading.Lock()
        self.diagnostics = {}
        self.last_arm_success_time = None
        self.last_hand_success_time = None

        self.cache = {
            "arm_state": None,
            "hand_state": None,
            "frames": None
        }

        print("[AsyncHW] Pre-fetching hardware states...")
        self._update_once()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print("[AsyncHW] Hardware background thread started.")

    def _update_once(self):
        started = time.perf_counter()
        try:
            arm_ok, arm_s = self.arm.get_joint_state()
            if not arm_ok:
                arm_s = None
        except:
            arm_s = None
        finally:
            self._record_diagnostic("arm_read", time.perf_counter() - started)

        started = time.perf_counter()
        try:
            hand_s = self.hand.get_composite_state()
        except:
            hand_s = None
        finally:
            self._record_diagnostic("hand_read", time.perf_counter() - started)

        started = time.perf_counter()
        try:
            frames = self.camera.get_frames()
        except:
            frames = None
        finally:
            self._record_diagnostic("camera_read", time.perf_counter() - started)

        if arm_s is not None and len(arm_s) == 7:
            self.last_arm_success_time = time.monotonic()
        if hand_s is not None:
            self.last_hand_success_time = time.monotonic()
        return arm_s, hand_s, frames

    def _record_diagnostic(self, name, elapsed_s):
        elapsed_ms = elapsed_s * 1000.0
        with self.diagnostics_lock:
            stat = self.diagnostics.setdefault(
                name, {"total_ms": 0.0, "max_ms": 0.0, "count": 0}
            )
            stat["total_ms"] += elapsed_ms
            stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
            stat["count"] += 1

    def consume_diagnostics(self):
        with self.diagnostics_lock:
            raw = self.diagnostics
            self.diagnostics = {}
        return {
            name: {
                "avg_ms": stat["total_ms"] / stat["count"],
                "max_ms": stat["max_ms"],
                "count": stat["count"],
            }
            for name, stat in raw.items()
            if stat["count"]
        }

    def _worker(self):
        while self.running:
            try:
                arm_s, hand_s, frames = self._update_once()

                with self.lock:
                    self.cache["arm_state"] = arm_s
                    self.cache["hand_state"] = hand_s
                    if frames is not None:
                        self.cache["frames"] = frames

                time.sleep(0.01)

            except Exception as e:
                print(f"[AsyncHW Error] {e}")
                time.sleep(0.1)

    def get_latest(self):
        with self.lock:
            return self.cache["arm_state"], self.cache["hand_state"], self.cache["frames"]

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


# ================= Server Class =================
class TeleopServer:
    def __init__(
        self,
        cmd_addr,
        img_addr,
        hand_backend,
        hand_device,
        flange_thickness,
        camera_mode,
        hand_effort_limit=None,
        kinect_addr=None,
        kinect_hand_eye_path=DEFAULT_KINECT_HAND_EYE_PATH,
        kinect_max_age_s=0.2,
        kinect_startup_timeout_s=30.0,
        kinect_pointcloud_points=8192,
        arm_backend="rm75", arm_ip=ROBOT_IP, arm_port=ROBOT_PORT, arm_side="A",
        marvin_sdk_root=None, marvin_config=None, hand_side="right",
        enable_motion=False, tool_xyz_m=(0,0,0), tool_rpy_deg=(0,0,0),
        tracking_to_base_rpy_deg=(0,0,0),
    ):
        self.running = True
        self.arm_backend = arm_backend
        self.motion_enabled = bool(enable_motion)
        self.hand_side = hand_side
        self.hand_model, self.hand_urdf, self.model_hand_limits = hand_spec(hand_side)
        self.arm_metadata = {"model": "MARVIN-M6-CCS-4.0" if arm_backend == "marvin" else "RM75",
                             "side": arm_side, "hand_side": hand_side,
                             "joint_units": "radian",
                             "tracking_to_base_rpy_deg": list(tracking_to_base_rpy_deg),
                             "tool_xyz_m": list(tool_xyz_m), "tool_rpy_deg": list(tool_rpy_deg)}
        self.tracking_to_base_rotation = R.from_euler("xyz", tracking_to_base_rpy_deg, degrees=True).as_matrix()
        self.max_linear_speed = 0.05 if arm_backend == "marvin" else MAX_EE_LINEAR_SPEED_M_S
        self.max_angular_speed = 20.0 if arm_backend == "marvin" else MAX_EE_ANGULAR_SPEED_DEG_S
        self.max_displacement = 0.15 if arm_backend == "marvin" else MAX_EE_DISPLACEMENT_M
        self.max_joint_error = 10.0 if arm_backend == "marvin" else MAX_JOINT_ERROR_DEG
        if (arm_backend == "marvin" or hand_side == "left") and kinect_addr:
            raise ValueError("MARVIN/left-hand Kinect geometry is not yet calibrated; use basic no-Kinect mode")
        if arm_backend == "marvin" and flange_thickness != 0:
            raise ValueError("MARVIN requires --flange-thickness 0; use --tool-xyz-m/--tool-rpy-deg")
        self.teleop_enabled = False
        self.last_command_time = None
        self.flange_thickness = flange_thickness
        self.camera_mode = camera_mode
        self.kinect_addr = kinect_addr
        self.kinect_max_age_ns = int(float(kinect_max_age_s) * 1e9)
        self.kinect_startup_timeout_s = float(kinect_startup_timeout_s)
        self.T_base_camera = None
        self.kinect_hand_eye_path = None
        if kinect_addr:
            if kinect_pointcloud_points < 1:
                raise ValueError(
                    "Kinect nearest-distance collection requires a positive "
                    "point-cloud capacity"
                )
            from utils.kinect_hand_eye import load_T_base_camera

            self.kinect_hand_eye_path = str(Path(kinect_hand_eye_path).resolve())
            self.T_base_camera = load_T_base_camera(self.kinect_hand_eye_path)
        self.distance_keypoint_specs = DEFAULT_KEYPOINT_SPECS
        self.distance_keypoint_names = tuple(
            spec["name"] for spec in self.distance_keypoint_specs
        )
        local_camera_names = {
            "both": ("D405", "ZED_M"),
            "d405": ("D405",),
            "zed": ("ZED_M",),
            "none": (),
        }[camera_mode]
        self.enabled_camera_names = local_camera_names + (
            ("KINECT",) if kinect_addr else ()
        )
        self.preview_camera = "ZED_M" if "ZED_M" in self.enabled_camera_names else (
            "D405" if "D405" in self.enabled_camera_names else (
                "KINECT" if "KINECT" in self.enabled_camera_names else "none"
            )
        )

        print("=" * 60)
        print(f"Teleop Server ({arm_backend} + Wuji {hand_side} Hand) Starting...")
        print("=" * 60)

        # ---------------- 1. Hardware Initialization ----------------

        # 1.1 Load kinematics before creating any hardware connection.
        self.keypoint_kinematics = None
        if arm_backend == "marvin":
            from utils.marvin_kinematics import MarvinKinematics, VENDOR_ROOT, DEFAULT_CONFIG
            from utils.marvin_controller import MarvinController
            marvin_sdk_root = marvin_sdk_root or VENDOR_ROOT
            marvin_config = marvin_config or DEFAULT_CONFIG
            self.kinematics = MarvinKinematics(marvin_config, arm_side, marvin_sdk_root,
                                               tool_xyz_m, tool_rpy_deg)
            self.arm_metadata["config_sha256"] = self.kinematics.model_config.sha256
            self.arm = MarvinController(arm_ip, side=arm_side, sdk_root=marvin_sdk_root,
                                        config_path=marvin_config, enable_motion=enable_motion)
            self.arm_metadata["controller_version"] = self.arm.version
        else:
            from utils.Realman_controller import RoboticArmController
            from utils.rm75_kinematics import RM75Kinematics
            self.kinematics = RM75Kinematics(URDF_PATH, base_pos=[0, 0, 0])
            self.arm = RoboticArmController(ip=arm_ip, port=arm_port)
        if kinect_addr:
            self.keypoint_kinematics = WujiFingertipKinematics()
        print(f"[Init] Arm={arm_backend}, hand={hand_side}, motion_allowed={self.motion_enabled}")

        # 1.3 Dexterous Hand
        print(f"[Init] Connecting to Wuji Hand with backend {hand_backend or 'wujihandpy'}...")
        try:
            self.hand = WujiHandDriver(
                device=hand_device,
                backend_spec=hand_backend,
                effort_limit=hand_effort_limit,
                side=hand_side, enable_motion=enable_motion, defer_enable=True,
            )

            hardware_limits = self.hand.get_joint_limits()
            self.hand_limits_min = (
                np.maximum(self.model_hand_limits[:, 0], hardware_limits[:, 0])
                + HAND_LIMIT_MARGIN
            )
            self.hand_limits_max = (
                np.minimum(self.model_hand_limits[:, 1], hardware_limits[:, 1])
                - HAND_LIMIT_MARGIN
            )
            if np.any(self.hand_limits_min >= self.hand_limits_max):
                raise RuntimeError(
                    "No valid Wuji joint range remains after intersecting the "
                    "URDF and hardware-calibrated limits"
                )
            device_info = self.hand.get_device_info()
            print(f"[Init] Wuji device info: {device_info}")
            # Preserve current pose; never send a HOME command during startup.
            self.hand.reset_tactile_sensors()
            print(f"[Init] Wuji Hand connected ({HAND_DOF} DoF).")
        except Exception as e:
            print(f"[Fatal] Wuji Hand connection failed: {e}")
            try:
                self.hand.close()
            except Exception:
                pass
            self.arm.disconnect()
            sys.exit(1)

        # 1.4 Cameras
        print(f"[Init] Camera mode: {self.camera_mode}")
        try:
            rs_reader = None
            zed_reader = None
            kinect_reader = None
            if "D405" in self.enabled_camera_names:
                from utils.Realsense_controller import MultiCameraReader

                print("[Init] Starting D405...")
                rs_reader = MultiCameraReader()
            if "ZED_M" in self.enabled_camera_names:
                from utils.Zed_controller_cv import ZEDCameraReader

                print("[Init] Starting ZED M...")
                zed_reader = ZEDCameraReader(fps=IMG_HZ)
            if "KINECT" in self.enabled_camera_names:
                from utils.kinect_stream import KinectZmqReceiver

                print(f"[Init] Connecting to Kinect publisher at {kinect_addr}...")
                kinect_reader = KinectZmqReceiver(
                    kinect_addr,
                    max_age_s=kinect_max_age_s,
                    pointcloud_capacity=kinect_pointcloud_points,
                )

            self.camera = CombinedCameraReader(rs_reader, zed_reader, kinect_reader)

            print("[Init] Starting Full Hardware Async Interface...")
            self.hw_interface = AsyncHardwareInterface(self.arm, self.hand, self.camera)

            frames = {}
            if self.enabled_camera_names:
                expected = ", ".join(self.enabled_camera_names)
                print(f"[Init] Waiting for first frame from: {expected}")
                wait_started = time.monotonic()
                while not all(name in frames for name in self.enabled_camera_names):
                    _, _, frames = self.hw_interface.get_latest()
                    frames = frames or {}
                    if (
                        "KINECT" in self.enabled_camera_names
                        and time.monotonic() - wait_started
                        > self.kinect_startup_timeout_s
                    ):
                        missing = sorted(set(self.enabled_camera_names) - set(frames))
                        raise TimeoutError(
                            f"Timed out waiting for camera frames: {missing}; "
                            "start kinect_foundationpose_publisher.py first"
                        )
                    time.sleep(0.1)
            else:
                print("[Init] Cameras disabled; control and non-image recording remain available.")

            if "ZED_M" in frames:
                print("[Init] ZED_M (Third-Person) Detected.")
            if "D405" in frames:
                print("[Init] D405 (First-Person) Detected.")
            if "KINECT" in frames:
                print("[Init] Azure Kinect synchronized publisher detected.")
        except Exception as e:
            print(f"[Fatal] Camera Init Failed: {e}")
            try:
                self.camera.stop()
            except Exception:
                pass
            self.arm.disconnect()
            self.hand.close()
            raise

        # 1.5 Data Collector
        self.collector = DataCollectorWuji(
            save_dir="data_records_wuji",
            hand_model=self.hand_model,
            hand_joint_names=JOINT_NAMES,
            hand_urdf=str(self.hand_urdf),
            arm_metadata=self.arm_metadata,
            flange_thickness=self.flange_thickness,
            camera_mode=self.camera_mode,
            camera_names=self.enabled_camera_names,
            distance_keypoint_names=self.distance_keypoint_names,
            distance_T_base_camera=self.T_base_camera,
        )

        self.record_queue = Queue(maxsize=30)
        self.record_thread = threading.Thread(target=self._record_worker, daemon=True)
        self.record_thread.start()

        self.stream_queue = Queue(maxsize=5)
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()

        # ---------------- 2. Network Initialization ----------------
        self.ctx = zmq.Context.instance()

        # Command Receiver (PULL from Client)
        self.cmd_sock = self.ctx.socket(zmq.PULL)
        self.cmd_sock.setsockopt(zmq.LINGER, 0)
        self.cmd_sock.setsockopt(zmq.RCVHWM, 1)  # Drop old commands if buffer full
        self.cmd_sock.bind(cmd_addr)
        print(f"[Net] Listening for commands at: {cmd_addr}")

        # Image Streamer (PUB to Client)
        self.img_sock = self.ctx.socket(zmq.PUB)
        self.img_sock.setsockopt(zmq.LINGER, 0)
        self.img_sock.bind(img_addr)
        print(f"[Net] Streaming images at: {img_addr}")

        # Vuer always needs an image buffer. In control-only mode it remains black.
        preview = self._build_preview_image(frames)
        self.img_shape = preview.shape if preview is not None else (720, 2560, 3)
        time.sleep(1.0)  # Wait for subscriber
        self._send_handshake()
        print(
            f"[Net] Handshake sent (camera_mode={self.camera_mode}, "
            f"preview={self.preview_camera}, shape={self.img_shape})"
        )

        # ---------------- 3. State Variables ----------------

        # Arm State
        success, q_init = self.arm.get_joint_state()
        if not success:
            print("[Error] Failed to read initial arm state!")
            self.cleanup()
            sys.exit(1)
        self.arm_curr_q_real_rad = np.deg2rad(np.array(q_init))
        self.arm_target_q_rad = self.arm_curr_q_real_rad.copy()

        # Hand State
        # Use new composite method for efficiency
        h_state = self.hand.get_composite_state()
        if h_state:
            self.hand_curr_q = h_state["joints"]
            self.hand_curr_tactile = h_state["tactile"]
        else:
            self.hand_curr_q = HOME_Q.copy()
            self.hand_curr_tactile = {}

        self.hand_target_q = self.hand_curr_q.copy()
        self.hand_command_q = self.hand_curr_q.copy()

        # Teleop Anchors
        self.human_anchor_pos = None
        self.human_anchor_rot = None
        self.robot_anchor_pos = None
        self.robot_anchor_rot = None
        self.desired_ee_pose = None
        self.last_valid_ee_pose = None  # [x,y,z, qx,qy,qz,qw]

        self.teleop_start_time = 0.0
        self.fixed_rot_mat = R.from_euler('z', FIXED_OFFSET_DEG, degrees=True).as_matrix()

    def run(self):
        """Main Control Loop with Performance Profiling"""
        dt = 1.0 / CONTROL_HZ
        img_dt = 1.0 / IMG_HZ

        last_loop_time = time.time()
        last_img_time = time.time()
        last_handshake_time = 0.0

        # Profiling Counters
        prof_last_print = time.time()
        prof_frame_count = 0
        prof_accum_sense = 0.0
        prof_accum_control = 0.0
        prof_accum_ik = 0.0
        prof_accum_arm_send = 0.0
        prof_accum_hand_send = 0.0
        prof_max_ik = 0.0
        prof_max_arm_send = 0.0
        prof_max_hand_send = 0.0
        prof_ik_failures = 0
        prof_last_ik_info = None
        prof_accum_copy = 0.0
        prof_accum_total = 0.0

        self.drop_counter = 0

        print(f"[System] Loop started at {CONTROL_HZ}Hz. Waiting for Client...")

        while self.running:
            t_start = time.time()

            # --- A. Receive Commands ---
            try:
                while True:
                    msg = self.cmd_sock.recv_json(flags=zmq.NOBLOCK)
                    self._process_msg(msg)
            except zmq.Again:
                pass

            self._expire_teleop_if_stale()

            # --- B. Control Loop ---
            if t_start - last_loop_time >= dt:

                current_frame_time = last_loop_time + dt
                # 1. Sense (Camera Read)
                t1 = time.time()
                obs_data = self._sense_and_update(t1)
                t_sense = time.time() - t1

                # 2. Control (computation plus synchronous hardware commands)
                t2 = time.perf_counter()
                control_cost = self._step_control(t1, obs_data)
                t_control = time.perf_counter() - t2

                # 3. Queue Record (Memory Copy)
                t3 = time.time()
                t_copy = 0.0
                if self.collector.recording:
                    copy_cost_ms = self._queue_record_data(t1, obs_data)
                    t_copy = copy_cost_ms / 1000.0

                t_total = time.time() - t_start
                last_loop_time += dt

                # Accumulate Stats
                prof_frame_count += 1
                prof_accum_sense += t_sense
                prof_accum_control += t_control
                prof_accum_ik += control_cost["ik_ms"] / 1000.0
                prof_accum_arm_send += control_cost["arm_send_ms"] / 1000.0
                prof_accum_hand_send += control_cost["hand_send_ms"] / 1000.0
                prof_max_ik = max(prof_max_ik, control_cost["ik_ms"])
                prof_max_arm_send = max(
                    prof_max_arm_send, control_cost["arm_send_ms"]
                )
                prof_max_hand_send = max(
                    prof_max_hand_send, control_cost["hand_send_ms"]
                )
                prof_ik_failures += int(control_cost["ik_failed"])
                if control_cost["ik_info"] is not None:
                    prof_last_ik_info = control_cost["ik_info"]
                prof_accum_copy += t_copy
                prof_accum_total += t_total

                if time.time() - prof_last_print >= 1.0:
                    duration = time.time() - prof_last_print

                    main_fps = prof_frame_count / duration

                    disk_fps = 0.0
                    if hasattr(self, 'rec_fps_counter'):
                        disk_fps = self.rec_fps_counter / duration
                        self.rec_fps_counter = 0  # Reset

                    drops = self.drop_counter
                    self.drop_counter = 0

                    avg_sense = (prof_accum_sense / prof_frame_count) * 1000
                    avg_control = (prof_accum_control / prof_frame_count) * 1000
                    avg_ik = (prof_accum_ik / prof_frame_count) * 1000
                    avg_arm_send = (prof_accum_arm_send / prof_frame_count) * 1000
                    avg_hand_send = (prof_accum_hand_send / prof_frame_count) * 1000
                    avg_copy = (prof_accum_copy / prof_frame_count) * 1000

                    rec_state = "[REC]" if self.collector.recording else "IDLE"

                    print(f"{rec_state} "
                          f"MainLoop: {main_fps:.1f}Hz | "
                          f"DiskWrite: {disk_fps:.1f}Hz | "
                          f"Drops: {drops} | "
                          f"Cost avg/max(ms): [SenseCache/FK:{avg_sense:.1f}] "
                          f"[Control:{avg_control:.1f}] "
                          f"[IK:{avg_ik:.1f}/{prof_max_ik:.1f}] "
                          f"[ArmSend:{avg_arm_send:.1f}/{prof_max_arm_send:.1f}] "
                          f"[HandSend:{avg_hand_send:.1f}/{prof_max_hand_send:.1f}] "
                          f"[Copy:{avg_copy:.1f}]",
                          flush=True)

                    hand_diag = self.hand.consume_diagnostics()
                    if hand_diag:
                        labels = {
                            "send_lock_wait": "SendLock",
                            "send_pdo": "SendPDO",
                            "send_total": "SendTotal",
                            "read_lock_wait": "ReadLock",
                            "read_pdo": "ReadPDO",
                            "read_total": "ReadTotal",
                            "heartbeat_sdo": "HeartbeatSDO",
                        }
                        details = []
                        for key, label in labels.items():
                            stat = hand_diag.get(key)
                            if stat is not None:
                                details.append(
                                    f"{label}:{stat['avg_ms']:.1f}/{stat['max_ms']:.1f}"
                                    f"#{stat['count']}"
                                )
                        # if details:
                        #     print(
                        #         "[WujiDiag] avg/max(ms)#count | " + " ".join(details),
                        #         flush=True,
                        #     )

                    hw_diag = self.hw_interface.consume_diagnostics()
                    hw_labels = {
                        "arm_read": "ArmRead",
                        "hand_read": "HandRead",
                        "camera_read": "CameraRead",
                    }
                    hw_details = []
                    for key, label in hw_labels.items():
                        stat = hw_diag.get(key)
                        if stat is not None:
                            hw_details.append(
                                f"{label}:{stat['avg_ms']:.1f}/{stat['max_ms']:.1f}"
                                f"#{stat['count']}"
                            )
                    # if hw_details:
                    #     print(
                    #         "[AsyncHWDiag] avg/max(ms)#count | "
                    #         + " ".join(hw_details),
                    #         flush=True,
                    #     )

                    if prof_last_ik_info is not None:
                        info = prof_last_ik_info
                        print(
                            f"[IKDiag] failures={prof_ik_failures}/{prof_frame_count} "
                            f"status={info.get('status', 'unknown')} "
                            f"iter={info.get('iter_count', 0)} "
                            f"pos_loss={info.get('pos', float('nan')):.6f} "
                            f"rot_loss={info.get('rot', float('nan')):.6f}",
                            flush=True,
                        )

                    slow_parts = {
                        "IK": prof_max_ik,
                        "ArmSend": prof_max_arm_send,
                        "HandSend": prof_max_hand_send,
                    }
                    bottleneck, slowest_ms = max(
                        slow_parts.items(), key=lambda item: item[1]
                    )
                    if slowest_ms > dt * 1000.0:
                        print(
                            f"[PerfWarn] {bottleneck} peaked at {slowest_ms:.1f} ms; "
                            f"the {CONTROL_HZ:.1f} Hz budget is {dt * 1000.0:.1f} ms.",
                            flush=True,
                        )

                    prof_last_print = time.time()
                    prof_frame_count = 0
                    prof_accum_sense = 0.0
                    prof_accum_control = 0.0
                    prof_accum_ik = 0.0
                    prof_accum_arm_send = 0.0
                    prof_accum_hand_send = 0.0
                    prof_max_ik = 0.0
                    prof_max_arm_send = 0.0
                    prof_max_hand_send = 0.0
                    prof_ik_failures = 0
                    prof_last_ik_info = None
                    prof_accum_copy = 0.0
                    prof_accum_total = 0.0

            # --- C. Stream Loop ---
            if t_start - last_img_time >= img_dt:
                if self.stream_queue.empty():
                    preview = self._build_preview_image(obs_data["images"])
                    if preview is not None:
                        self.stream_queue.put((t_start, preview.copy()))
                last_img_time += img_dt

            # PUB has no request/response handshake. Repeat metadata so a client
            # started later can still connect, including when cameras are disabled.
            if t_start - last_handshake_time >= 1.0:
                self._send_handshake()
                last_handshake_time = t_start

            time.sleep(0.0001)

    def _record_worker(self):
        self.rec_fps_counter = 0
        while self.running or not self.record_queue.empty():
            try:
                data = self.record_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if data is not None:
                    self.collector.add_frame(data)
                    self.rec_fps_counter += 1
            except Exception as exc:
                print(f"[DataCollector] Frame append failed: {exc}", flush=True)
            finally:
                self.record_queue.task_done()

    def _stream_worker(self):
        while self.running:
            try:
                item = self.stream_queue.get(timeout=0.1)
                if item is None: continue

                now, preview_rgb = item
                preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
                ok, encoded_img = cv2.imencode('.jpg', preview_bgr, encode_param)
                if not ok:
                    continue

                meta = self._image_metadata(int(now * 1000), preview_rgb.shape)
                meta["format"] = "jpg"
                self.img_sock.send_multipart([
                    json.dumps(meta).encode("utf-8"),
                    encoded_img.tobytes()
                ])
            except Exception:
                pass

    def _image_metadata(self, seq, shape):
        return {
            "seq": seq,
            "net": True,
            "shape": tuple(shape),
            "dtype": "uint8",
            "shm": "None",
            "camera_mode": self.camera_mode,
            "preview_camera": self.preview_camera,
        }

    def _send_handshake(self):
        self.img_sock.send_json(self._image_metadata(0, self.img_shape))

    def _build_preview_image(self, frames):
        frames = frames or {}
        if self.preview_camera == "ZED_M" and "ZED_M" in frames:
            return np.hstack(
                (frames["ZED_M"]["color"], frames["ZED_M"]["color_right"])
            )
        if self.preview_camera == "D405" and "D405" in frames:
            color_rgb = cv2.cvtColor(frames["D405"]["color"], cv2.COLOR_BGR2RGB)
            return np.hstack((color_rgb, color_rgb))
        if self.preview_camera == "KINECT" and "KINECT" in frames:
            color_rgb = cv2.cvtColor(
                frames["KINECT"]["color"], cv2.COLOR_BGR2RGB
            )
            return np.hstack((color_rgb, color_rgb))
        return None

    def _sense_and_update(self, timestamp):
        arm_state, hand_comp, cam_frames = self.hw_interface.get_latest()
        cam_frames = cam_frames or {}
        control_monotonic_ns = time.monotonic_ns()

        # Detect missing feedback even when previous numeric values remain cached.
        now = time.monotonic()
        arm_stamp = self.hw_interface.last_arm_success_time
        hand_stamp = self.hw_interface.last_hand_success_time
        if self.teleop_enabled and (arm_stamp is None or hand_stamp is None
            or now-arm_stamp > 0.25 or now-hand_stamp > 0.25):
            self._stop_teleop("Hardware feedback stale")
        # 1. Arm State
        if arm_state:
            self.arm_curr_q_real_rad = np.deg2rad(np.array(arm_state))

        # 2. Hand State
        if hand_comp:
            self.hand_curr_q = hand_comp["joints"]
            self.hand_curr_tactile = hand_comp["tactile"]

        # 3. Camera
        if "KINECT" in cam_frames:
            kinect_frame = dict(cam_frames["KINECT"])
            age_ns = (
                control_monotonic_ns - kinect_frame["capture_monotonic_ns"]
            )
            kinect_frame["sync_age_ms"] = age_ns / 1e6
            kinect_frame["valid"] = bool(
                kinect_frame.get("sample_valid", kinect_frame["pose_valid"])
                and 0 <= age_ns <= self.kinect_max_age_ns
            )
            cam_frames = dict(cam_frames)
            cam_frames["KINECT"] = kinect_frame

        # 4. FK
        ee_pos, ee_quat = self.kinematics.compute_fk(
            self.arm_curr_q_real_rad, flange_thickness=self.flange_thickness
        )
        ee_pose = np.concatenate([ee_pos, ee_quat])

        return {
            "time": timestamp,
            "monotonic_time_ns": control_monotonic_ns,
            "arm_qpos": self.arm_curr_q_real_rad,
            "hand_qpos": self.hand_curr_q,
            "tactile": self.hand_curr_tactile,
            "images": cam_frames,
            "ee_pose": ee_pose
        }

    def _step_control(self, now, obs):
        """
        Calculate and execute control actions.
        """
        costs = {
            "ik_ms": 0.0,
            "arm_send_ms": 0.0,
            "hand_send_ms": 0.0,
            "ik_failed": 0,
            "ik_info": None,
        }
        err_deg = 0.0
        fps = 0.0

        # 1. Safety Check (Arm Lag)
        err_deg = np.rad2deg(np.max(np.abs(obs["arm_qpos"] - self.arm_target_q_rad)))

        current_limit = getattr(self, "max_joint_error", MAX_JOINT_ERROR_DEG)

        if self.teleop_enabled:
            if err_deg > current_limit:
                print(f"\n[DANGER] Arm Lag > {current_limit}° ({err_deg:.1f}°). Resetting target.")
                self._stop_teleop("Arm tracking error")
                return costs

            # 2. Inverse Kinematics (Arm)
            if self.desired_ee_pose is not None and self.last_valid_ee_pose is not None:
                target_pose = limit_pose_step(
                    self.last_valid_ee_pose,
                    self.desired_ee_pose,
                    getattr(self, "max_linear_speed", MAX_EE_LINEAR_SPEED_M_S) / CONTROL_HZ,
                    np.deg2rad(getattr(self, "max_angular_speed", MAX_EE_ANGULAR_SPEED_DEG_S) / CONTROL_HZ),
                )
                tgt_pos = target_pose[:3]
                tgt_quat = target_pose[3:]

                t0 = time.perf_counter()
                # Use new solver
                ik_q_rad = self.kinematics.solve_ik(
                    obs["arm_qpos"],
                    tgt_pos,
                    tgt_quat,
                    flange_thickness=self.flange_thickness,
                    initial_q=self.arm_target_q_rad,
                )
                costs["ik_ms"] = (time.perf_counter() - t0) * 1000
                costs["ik_info"] = dict(self.kinematics.last_solve_info)
                ik_success = bool(self.kinematics.last_solve_info["success"])
                costs["ik_failed"] = int(not ik_success)

                if ik_success:
                    # Smooth Interpolation
                    next_q_rad = self.arm_target_q_rad * (1 - ARM_SMOOTH_ALPHA) + ik_q_rad * ARM_SMOOTH_ALPHA

                    # Execute Arm Command
                    t0 = time.perf_counter()
                    try:
                        self.arm.move_j_stream(np.rad2deg(next_q_rad))
                    except Exception as exc:
                        self._stop_teleop(f"Arm command failed: {exc}")
                        return costs
                    costs["arm_send_ms"] = (time.perf_counter() - t0) * 1000
                    self.arm_target_q_rad = next_q_rad
                    next_pos, next_quat = self.kinematics.compute_fk(
                        next_q_rad, flange_thickness=self.flange_thickness
                    )
                    self.last_valid_ee_pose = np.concatenate(
                        [next_pos, next_quat]
                    )
                elif getattr(self, "arm_backend", "rm75") == "marvin":
                    self._stop_teleop("MARVIN inverse kinematics rejected target")
                    return costs

            # 3. Hand Control
            # Smooth Hand
            next_hand_command = (
                self.hand_command_q * (1 - HAND_SMOOTH_ALPHA)
                + self.hand_target_q * HAND_SMOOTH_ALPHA
            )
            safe_joints = np.clip(
                next_hand_command, self.hand_limits_min, self.hand_limits_max
            )
            try:
                t0 = time.perf_counter()
                self.hand.send_joints(safe_joints)
                costs["hand_send_ms"] = (time.perf_counter() - t0) * 1000
                self.hand_command_q = safe_joints
            except Exception as exc:
                costs["hand_send_ms"] = (time.perf_counter() - t0) * 1000
                print(f"\n[DANGER] Wuji Hand control failed: {exc}. Teleoperation stopped.")
                self._stop_teleop(f"Hand command failed: {exc}")
                self.arm_target_q_rad = obs["arm_qpos"].copy()

        return costs

    def _queue_record_data(self, timestamp, obs):
        t0 = time.time()
        distances = self._estimate_keypoint_distances(
            obs["arm_qpos"], obs["hand_qpos"], obs["images"]
        )
        img_data = {}
        kinect_data = None
        for cam_name, frames in obs["images"].items():
            rgb_right_img = None
            if cam_name in ("D405", "KINECT"):
                rgb_img = cv2.cvtColor(frames["color"], cv2.COLOR_BGR2RGB)
            else:
                rgb_img = frames["color"]
                rgb_right_img = frames.get("color_right")
            depth_img = frames["depth"]

            if cam_name == "ZED_M":
                h, w = rgb_img.shape[:2]
                new_size = (w // 2, h // 2)

                rgb_img = cv2.resize(rgb_img, new_size, interpolation=cv2.INTER_LINEAR)
                if rgb_right_img is not None:
                    rgb_right_img = cv2.resize(
                        rgb_right_img, new_size, interpolation=cv2.INTER_LINEAR
                    )

                depth_img = cv2.resize(depth_img, new_size, interpolation=cv2.INTER_NEAREST)
            img_data[cam_name] = {
                "rgb": rgb_img.copy(),
                "depth": depth_img.copy()
            }
            if rgb_right_img is not None:
                img_data[cam_name]["rgb_right"] = rgb_right_img.copy()

            if cam_name == "KINECT":
                kinect_data = {
                    "tracking_mode": np.bytes_(
                        frames.get("tracking_mode", "foundationpose")
                    ),
                    "valid": np.uint8(bool(frames["valid"])),
                    "sample_valid": np.uint8(
                        bool(frames.get("sample_valid", frames["pose_valid"]))
                    ),
                    "pose_valid": np.uint8(bool(frames["pose_valid"])),
                    "segmentation_valid": np.uint8(
                        bool(frames.get("segmentation_valid", False))
                    ),
                    "clouds_updated": np.uint8(bool(frames["clouds_updated"])),
                    "frame_id": np.int64(frames["frame_id"]),
                    "capture_monotonic_ns": np.int64(
                        frames["capture_monotonic_ns"]
                    ),
                    "capture_wall_ns": np.int64(frames["capture_wall_ns"]),
                    "receive_monotonic_ns": np.int64(
                        frames["receive_monotonic_ns"]
                    ),
                    "sync_age_ms": np.float32(frames["sync_age_ms"]),
                    "inference_ms": np.float32(frames["inference_ms"]),
                    "intrinsics": np.asarray(
                        frames["intrinsics"], dtype=np.float32
                    ).copy(),
                    "object_pose_camera": np.asarray(
                        frames["object_pose_camera"], dtype=np.float32
                    ).copy(),
                    "pointcloud_count": np.int32(frames["pointcloud_count"]),
                }
                pointcloud_xyz = np.asarray(
                    frames["pointcloud_xyz"], dtype=np.float32
                )
                pointcloud_rgb = np.asarray(
                    frames["pointcloud_rgb"], dtype=np.uint8
                )
                if pointcloud_xyz.size:
                    kinect_data["pointcloud_xyz"] = pointcloud_xyz.copy()
                    kinect_data["pointcloud_rgb"] = pointcloud_rgb.copy()

        frame_data = {
            "time": timestamp,
            "monotonic_time_ns": np.int64(obs.get("monotonic_time_ns", 0)),
            "observations": {
                "qpos": np.concatenate([obs["arm_qpos"], obs["hand_qpos"]]),
                "ee_pose": obs["ee_pose"],
                "distances": distances,
                "images": img_data,
                "tactile": obs["tactile"]
            },
            "action": {
                "arm_target_q": self.arm_target_q_rad,
                "hand_target_q": self.hand_target_q,
                "hand_command_q": self.hand_command_q,
                "ee_target_pose": self.last_valid_ee_pose if self.last_valid_ee_pose is not None else np.zeros(7)
            }
        }
        if kinect_data is not None:
            frame_data["observations"]["kinect"] = kinect_data
        copy_cost = (time.time() - t0) * 1000

        try:
            self.record_queue.put_nowait(frame_data)
        except:
            if not hasattr(self, 'drop_counter'): self.drop_counter = 0
            self.drop_counter += 1
            print("[Warn] Record Queue Full! Dropping frame.")

        return copy_cost

    def _estimate_keypoint_distances(self, arm_q_rad, hand_q_rad, camera_frames):
        """Return each hand keypoint's nearest target-cloud distance in meters."""
        if getattr(self, "keypoint_kinematics", None) is None:
            return np.full(len(self.distance_keypoint_names), np.nan, np.float64)
        positions = self.keypoint_kinematics.estimate_keypoints(
            arm_q_rad,
            hand_q_rad,
            self.flange_thickness,
            keypoint_specs=self.distance_keypoint_specs,
            arm_angle_unit="radian",
        )
        keypoints_base = np.asarray(
            [positions[name] for name in self.distance_keypoint_names],
            dtype=np.float64,
        )
        invalid = np.full(len(self.distance_keypoint_names), np.nan, np.float64)
        frame = (camera_frames or {}).get("KINECT")
        if frame is None or not frame.get("valid", False) or self.T_base_camera is None:
            return invalid

        points_camera = np.asarray(frame.get("pointcloud_xyz"), dtype=np.float64)
        point_count = int(frame.get("pointcloud_count", 0))
        if (
            points_camera.ndim != 2
            or points_camera.shape[1:] != (3,)
            or point_count <= 0
            or point_count > len(points_camera)
        ):
            return invalid

        from utils.kinect_hand_eye import keypoint_to_camera_cloud_distances

        return keypoint_to_camera_cloud_distances(
            keypoints_base,
            points_camera[:point_count],
            self.T_base_camera,
        )

    def _record_data(self, timestamp, obs):
        """
        Pack observations and actions into the DataCollector.
        """
        if not self.collector.recording:
            return

        # Prepare Image Data (Extract arrays from dict)
        # Note: We only store raw arrays to save RAM, conversion to h5 handles structure
        img_data = {}
        for cam_name, frames in obs["images"].items():
            img_data[cam_name] = {
                "rgb": frames["color"].copy(),
                "depth": frames["depth"].copy()
            }

        frame_data = {
            "time": timestamp,
            "observations": {
                "qpos": np.concatenate([obs["arm_qpos"], obs["hand_qpos"]]),
                "ee_pose": obs["ee_pose"],  # [x,y,z,qx,qy,qz,qw]
                "images": img_data,
                "tactile": obs["tactile"]  # Nested dict
            },
            "action": {
                "arm_target_q": self.arm_target_q_rad,
                "hand_target_q": self.hand_target_q,
                "hand_command_q": self.hand_command_q,
                "ee_target_pose": self.last_valid_ee_pose if self.last_valid_ee_pose is not None else np.zeros(7)
            }
        }

        self.collector.add_frame(frame_data)

    def _process_msg(self, msg):
        """Handle ZMQ commands from Client."""
        msg_type = msg.get("type", "cmd")
        action = msg.get("action")

        # 1. Control Logic Switches
        if msg_type == "control":
            if action == "teleop_on":
                if msg.get("hand_side", "right") != getattr(self, "hand_side", "right"):
                    print("[Warn] Client/server hand sides differ; refusing teleoperation")
                    return
                self._enable_teleop()

            elif action == "teleop_off":
                self._stop_teleop("Client paused")
                self.teleop_enabled = False
                self.last_command_time = None
                self.human_anchor_pos = None
                self.human_anchor_rot = None
                print("\n[Server] Teleop OFF")

            elif action == "start_recording":
                task_name = msg.get("task_name", "default")
                self.collector.start_recording(task_name, msg.get("input_device", "unknown"))

            elif action == "stop_recording":
                self._finish_recording()

            elif action == "quit":
                self.running = False

        # 2. Motion Commands (Only when Teleop Active)
        elif msg_type == "cmd" and self.teleop_enabled:
            # Update Arm Target (Cartesian)
            if msg.get("hand_side", "right") != getattr(self, "hand_side", "right"):
                self._stop_teleop("Client hand side changed")
                return
            raw_wrist = msg.get("wrist_mat", msg.get("right_wrist_mat"))
            try:
                wrist = np.asarray(raw_wrist, dtype=np.float64)
            except (TypeError, ValueError):
                wrist = np.empty(0)
            if wrist.shape != (16,) or not np.isfinite(wrist).all():
                print("[Warn] Ignoring invalid wrist command")
                return
            wrist_matrix = wrist.reshape(4, 4)
            rotation = wrist_matrix[:3, :3]
            if (
                not np.allclose(wrist_matrix[3], [0, 0, 0, 1], atol=1e-3)
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-2)
                or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-2)
            ):
                print("[Warn] Ignoring invalid wrist command")
                return

            # Update Hand Target (Joints)
            raw_hand = msg.get("hand_qpos")
            if raw_hand is not None:
                try:
                    hand_target = validate_joint_vector(raw_hand, label="Wuji network target")
                except ValueError as exc:
                    print(f"[Warn] Ignoring invalid Wuji command: {exc}")
                    return
                self._update_arm_target_from_wrist(wrist)
                self.hand_target_q = hand_target
                self.last_command_time = time.monotonic()

    def _expire_teleop_if_stale(self):
        if (
            self.teleop_enabled
            and self.last_command_time is not None
            and time.monotonic() - self.last_command_time > COMMAND_WATCHDOG_S
        ):
            self._stop_teleop("Input command watchdog expired")
            self.teleop_enabled = False
            self.last_command_time = None
            self.human_anchor_pos = None
            self.human_anchor_rot = None
            self._finish_recording()

    def _stop_teleop(self, reason):
        self.teleop_enabled = False
        self.last_command_time = None
        self.human_anchor_pos = None
        self.human_anchor_rot = None
        if hasattr(getattr(self, "arm", None), "stop"):
            self.arm.stop()
        print(f"[Stop] {reason}; press T again after resolving the cause")

    def _enable_teleop(self):
        """
        Initialize Teleoperation:
        1. Read current robot state.
        2. Compute current TCP pose (Anchor).
        3. Set safety timers.
        """
        if not getattr(self, "motion_enabled", False):
            print("[Read-only] Start the server with --enable-motion to permit movement")
            return
        if self.teleop_enabled:
            return
        # Read State
        ok, state = self.arm.get_joint_state()
        if not ok:
            state = None
        if not state:
            print("[Warn] Cannot read arm state, Teleop enable failed.")
            return

        real_q = np.deg2rad(np.array(state))
        self.arm_curr_q_real_rad = real_q
        self.arm_target_q_rad = real_q.copy()

        # Compute FK Anchor using new solver
        anchor_pos, anchor_quat = self.kinematics.compute_fk(
            real_q, flange_thickness=self.flange_thickness
        )

        self.robot_anchor_pos = anchor_pos
        self.robot_anchor_rot = R.from_quat(anchor_quat).as_matrix()
        self.human_anchor_pos = None  # Will be set on first received packet
        self.human_anchor_rot = None
        anchor_pose = np.concatenate([anchor_pos, anchor_quat])
        self.desired_ee_pose = anchor_pose.copy()
        self.last_valid_ee_pose = anchor_pose.copy()

        # Reset Hand
        try:
            h_curr = self.hand.get_current_joints()
        except Exception as exc:
            print(f"[Warn] Cannot read Wuji Hand state; teleop enable failed: {exc}")
            return
        if h_curr is not None:
            self.hand_curr_q = h_curr
            self.hand_target_q = h_curr.copy()
            self.hand_command_q = h_curr.copy()

        if hasattr(self.arm, "enable"):
            try:
                self.hand.enable()
                self.arm.enable()
            except Exception as exc:
                self._stop_teleop(f"Enable failed: {exc}")
                self.hand.close()
                self.running = False
                return
        else:
            self.hand.enable()
        self.teleop_enabled = True
        self.last_command_time = time.monotonic()
        self.teleop_start_time = time.time()
        print(f"\n[Server] Teleop ON. Robot Anchor: {self.robot_anchor_pos}")

    def _update_arm_target_from_wrist(self, flat_mat):
        """
        Map Human Wrist Matrix -> Robot TCP Pose (Delta Mapping)
        """
        if self.robot_anchor_pos is None:
            return

        mat = np.array(flat_mat).reshape(4, 4)
        basis = getattr(self, "tracking_to_base_rotation", np.eye(3))
        human_pos = basis @ mat[:3, 3]
        human_rot_mat = basis @ self.fixed_rot_mat @ mat[:3, :3]

        # Initialize Human Anchor on first frame
        if self.human_anchor_pos is None:
            self.human_anchor_pos = human_pos.copy()
            self.human_anchor_rot = human_rot_mat.copy()
            return

        # 1. Position Mapping (Delta)
        delta_human = human_pos - self.human_anchor_pos
        delta_robot = delta_human * ARM_MOTION_SCALE
        delta_norm = np.linalg.norm(delta_robot)
        max_displacement = getattr(self, "max_displacement", MAX_EE_DISPLACEMENT_M)
        if delta_norm > max_displacement:
            delta_robot *= max_displacement / delta_norm
        target_pos = self.robot_anchor_pos + delta_robot

        # Safety: Virtual Table
        if getattr(self, "arm_backend", "rm75") != "marvin" and target_pos[2] < Z_MIN_HEIGHT:
            target_pos[2] = Z_MIN_HEIGHT

        # 2. Rotation Mapping (relative to both activation poses)
        human_delta_rot = human_rot_mat @ self.human_anchor_rot.T
        target_rot_mat = human_delta_rot @ self.robot_anchor_rot
        target_quat = R.from_matrix(target_rot_mat).as_quat()

        self.desired_ee_pose = np.concatenate([target_pos, target_quat])

    def _finish_recording(self):
        if not self.collector.recording:
            return
        # The control loop is the only producer and calls this method itself, so
        # no new recording frames are queued while the existing writes drain.
        self.record_queue.join()
        self.collector.stop_recording()

    def cleanup(self):
        print("\n[Exit] Cleaning up resources...")
        if getattr(self, "_cleaned_up", False):
            return
        self._cleaned_up = True
        self.running = False
        if hasattr(self.arm, "stop"):
            self.arm.stop()
        # Save data if still recording
        if self.collector.recording:
            self._finish_recording()
        self.running = False

        try:
            self.hw_interface.stop()
        except Exception:
            pass
        try:
            self.arm.disconnect()
        except:
            pass
        try:
            self.hand.close()
        except:
            pass
        try:
            self.camera.stop()
        except:
            pass

        self.cmd_sock.close()
        self.img_sock.close()
        self.ctx.term()


def main():
    parser = argparse.ArgumentParser(description="Teleop Server (Wuji Data Collection Mode)")
    from utils.marvin_kinematics import VENDOR_ROOT, DEFAULT_CONFIG
    parser.add_argument("--arm-backend", choices=("rm75", "marvin"), default="rm75")
    parser.add_argument("--arm-ip", default=ROBOT_IP)
    parser.add_argument("--arm-port", type=int, default=ROBOT_PORT)
    parser.add_argument("--arm-side", choices=("A", "B"), default="A")
    parser.add_argument("--marvin-sdk-root", type=Path, default=VENDOR_ROOT)
    parser.add_argument("--marvin-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--enable-motion", action="store_true", help="Allow arm/hand motion; default is read-only")
    parser.add_argument("--tool-xyz-m", type=float, nargs=3, default=(0,0,0))
    parser.add_argument("--tool-rpy-deg", type=float, nargs=3, default=(0,0,0))
    parser.add_argument("--tracking-to-base-rpy-deg", type=float, nargs=3, default=(0,0,0))
    parser.add_argument("--cmd-addr", default="tcp://*:6100", help="ZMQ Command Bind Address")
    parser.add_argument("--img-addr", default="tcp://*:6101", help="ZMQ Image Stream Bind Address")
    parser.add_argument(
        "--camera-mode",
        choices=("both", "d405", "zed", "none"),
        default="none",
        help=(
            "Camera hardware/data selection: both, d405 only, zed only, or "
            "none for arm/hand-only operation"
        ),
    )
    parser.add_argument(
        "--kinect-addr",
        default=None,
        help=(
            "Optional Azure Kinect/FoundationPose publisher address, e.g. "
            "tcp://127.0.0.1:6200; can be combined with any --camera-mode"
        ),
    )
    parser.add_argument(
        "--kinect-max-age",
        type=float,
        default=0.2,
        help="Mark a synchronized Kinect sample invalid after this many seconds",
    )
    parser.add_argument(
        "--kinect-startup-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for the first Kinect publisher sample",
    )
    parser.add_argument(
        "--kinect-pointcloud-points",
        type=int,
        default=1024,
        help="Fixed HDF5 point-cloud capacity; 0 disables point arrays",
    )
    parser.add_argument(
        "--hand-backend",
        default="wujihandpy",
        help=(
            "Hand backend: wujihandpy (default), mock (offline only), or a "
            "python.module:BackendClass adapter"
        ),
    )
    parser.add_argument(
        "--hand-device",
        default=None,
        help="Wuji USB serial number; omitted selects --hand-side",
    )
    parser.add_argument(
        "--hand-effort-limit",
        type=float,
        default=0.6,
        metavar="AMPERES",
        help=(
            "Maximum motor current for every Wuji joint in amperes; "
            f"must be within (0, {MAX_JOINT_EFFORT_LIMIT_A}] and is left "
            "unchanged when omitted"
        ),
    )
    parser.add_argument(
        "--flange-thickness",
        type=float,
        default=DEFAULT_FLANGE_THICKNESS,
        help="RM75 link_7 to Wuji TCP offset along local Z, in meters",
    )
    parser.add_argument(
        "--kinect-hand-eye",
        type=Path,
        default=DEFAULT_KINECT_HAND_EYE_PATH,
        help=(
            "JSON hand-eye calibration containing T_base_camera; required when "
            "--kinect-addr is enabled"
        ),
    )
    args = parser.parse_args()
    if args.kinect_max_age <= 0 or args.kinect_startup_timeout <= 0:
        parser.error("Kinect age and startup timeout must be positive")
    if args.kinect_pointcloud_points < 0:
        parser.error("--kinect-pointcloud-points cannot be negative")
    if args.kinect_addr and args.kinect_pointcloud_points < 1:
        parser.error(
            "--kinect-pointcloud-points must be positive when --kinect-addr is enabled"
        )
    if args.hand_effort_limit is not None and (
        not np.isfinite(args.hand_effort_limit)
        or args.hand_effort_limit <= 0.0
        or args.hand_effort_limit > MAX_JOINT_EFFORT_LIMIT_A
    ):
        parser.error(
            "--hand-effort-limit must be within "
            f"(0, {MAX_JOINT_EFFORT_LIMIT_A}] amperes"
        )

    if not np.isfinite([*args.tool_xyz_m, *args.tool_rpy_deg, *args.tracking_to_base_rpy_deg]).all():
        parser.error("Tool and tracking transforms must be finite")
    if args.arm_backend == "marvin" and args.flange_thickness != 0:
        parser.error("MARVIN requires --flange-thickness 0; specify tool transform instead")
    if (args.arm_backend == "marvin" or args.hand_side == "left") and args.kinect_addr:
        parser.error("MARVIN/left-hand Kinect is not supported in this basic teleoperation path")
    server = TeleopServer(
        args.cmd_addr,
        args.img_addr,
        hand_backend=args.hand_backend,
        hand_device=args.hand_device,
        hand_effort_limit=args.hand_effort_limit,
        flange_thickness=args.flange_thickness,
        camera_mode=args.camera_mode,
        arm_backend=args.arm_backend, arm_ip=args.arm_ip, arm_port=args.arm_port,
        arm_side=args.arm_side, marvin_sdk_root=args.marvin_sdk_root,
        marvin_config=args.marvin_config, hand_side=args.hand_side,
        enable_motion=args.enable_motion, tool_xyz_m=args.tool_xyz_m,
        tool_rpy_deg=args.tool_rpy_deg, tracking_to_base_rpy_deg=args.tracking_to_base_rpy_deg,
        kinect_addr=args.kinect_addr,
        kinect_hand_eye_path=args.kinect_hand_eye,
        kinect_max_age_s=args.kinect_max_age,
        kinect_startup_timeout_s=args.kinect_startup_timeout,
        kinect_pointcloud_points=args.kinect_pointcloud_points,
    )

    # Signal Handler for Ctrl+C
    def signal_handler(sig, frame):
        server.running = False

    signal.signal(signal.SIGINT, signal_handler)

    try:
        server.run()
    except Exception as e:
        print(f"[Fatal Crash] {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        server.cleanup()


if __name__ == "__main__":
    main()
