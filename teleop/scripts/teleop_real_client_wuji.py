#!/usr/bin/env python3
"""
Teleoperation Client for RM75 + Wuji Hand data collection
Responsibility:
1. Receive VR hand tracking data from Vuer.
2. Perform Motion Retargeting (Human Hand -> Robot Hand).
3. Send Action Commands to the Server via ZMQ.
4. Manage Recording State (Signal Server).
"""

import argparse
import json
import time
import threading
import sys
import numpy as np
import zmq
import traceback
import cv2
from multiprocessing import shared_memory, Event
from queue import Queue, Empty
from pathlib import Path

TELEOP_ROOT = Path(__file__).resolve().parents[1]
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

try:
    from utils.wuji_hand_spec import JOINT_NAMES
except ImportError:
    print("[Error] Wuji teleoperation dependencies could not be imported.")
    traceback.print_exc()
    sys.exit(1)


class VuerTeleop:
    """
    Handles communication with the Vuer VR client via Shared Memory.
    Adapts to dynamic resolutions provided by the server.
    """

    def __init__(self, shm_name: str = None, img_shape_override: tuple = None):
        original_argv = sys.argv[:]
        try:
            # Vuer's params_proto parses sys.argv at import time. Keep its CLI
            # separate from this client's argparse options.
            sys.argv = [sys.argv[0]]
            from teleop.TeleVision import OpenTeleVision
            from teleop.Preprocessor_wuji import VuerPreprocessorWuji
        finally:
            sys.argv = original_argv

        # 1. Dynamic Resolution Adaptation
        if img_shape_override is not None:
            h, w_total = img_shape_override[0], img_shape_override[1]
            # Assuming side-by-side stereo, single eye width is w_total / 2
            self.resolution = (h, w_total // 2)
            print(f"[VuerTeleop] Resolution adapted from server: {self.resolution}")
        else:
            self.resolution = (720, 1280)
            print(f"[VuerTeleop] Using default resolution: {self.resolution}")

        # Crop settings (if needed in future)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (
            self.resolution[0] - self.crop_size_h,
            self.resolution[1] - 2 * self.crop_size_w
        )

        # Final Buffer Shape
        self.img_shape = img_shape_override or (
            self.resolution_cropped[0],
            2 * self.resolution_cropped[1],
            3
        )

        # 2. Shared Memory Setup
        # Case A: Local Simulation or Existing SHM (shm_name provided)
        if shm_name and shm_name != "None":
            try:
                self.shm = shared_memory.SharedMemory(name=shm_name)
                print(f"[VuerTeleop] Attached to existing SHM: {shm_name}")
            except FileNotFoundError as e:
                raise RuntimeError(
                    f"Could not find SHM '{shm_name}'. Ensure Server is running."
                ) from e
            self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.shm.buf)

        # Case B: Network Mode or Standalone Test (Create local SHM)
        else:
            print(f"[VuerTeleop] Network mode: Creating local buffer ({self.img_shape})")
            try:
                self.shm = shared_memory.SharedMemory(
                    create=True,
                    size=np.prod(self.img_shape) * np.uint8().itemsize
                )
            except FileExistsError:
                # Fallback if name collision or cleanup failed
                self.shm = shared_memory.SharedMemory(
                    create=True,
                    size=np.prod(self.img_shape) * np.uint8().itemsize
                )

            self.img_array = np.ndarray(self.img_shape, dtype=np.uint8, buffer=self.shm.buf)
            self.img_array[:] = 0  # Initialize black

        # 3. OpenTeleVision Interface
        self.image_queue = Queue()
        self.toggle_streaming = Event()

        # Initialize Vuer backend
        self.tv = OpenTeleVision(
            self.resolution_cropped,
            self.shm.name,
            self.image_queue,
            self.toggle_streaming,
            ngrok=True
        )
        self.processor = VuerPreprocessorWuji()

    def step(self):
        """Retrieve latest hand tracking data from VR."""
        return self.processor.process(self.tv, "wuji")

    def close(self):
        """Clean up resources."""
        try:
            self.shm.close()
            # self.shm.unlink() # Uncomment if you want to strictly destroy the SHM file
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Teleop Client (Lightweight Control Only)")
    parser.add_argument("--cmd-addr", default="tcp://127.0.0.1:6100", help="ZMQ Command PUSH Address")
    parser.add_argument("--img-addr", default="tcp://127.0.0.1:6101", help="ZMQ Image Metadata SUB Address")
    parser.add_argument(
        "--camera-mode",
        choices=("both", "d405", "zed", "none"),
        default="none",
        help="Expected server camera mode; must match the server --camera-mode",
    )
    parser.add_argument("--teleop-hz", type=float, default=60.0, help="Target Control Frequency (Hz)")
    parser.add_argument("--devise", "--device", dest="input_device", choices=("vr", "glove"), default="vr", help="VR or Wuji Glove + Vive Tracker input")
    parser.add_argument("--retarget-config", default="teleop/wuji.yml", help="Path to Wuji retargeting config (VR only)")
    parser.add_argument("--hand-side", choices=("left", "right"), default="right")
    parser.add_argument("--tracker-source", choices=("openvr", "udp"), default="openvr")
    parser.add_argument("--tracker-bind", default="0.0.0.0")
    parser.add_argument("--tracker-port", type=int, default=6210)
    parser.add_argument("--tracker-source-ip", help="Windows sender IPv4; required for UDP")
    parser.add_argument("--glove-sn", help="Wuji Glove serial number")
    parser.add_argument("--tracker-serial", help="Vive Tracker serial number")
    parser.add_argument("--glove-user-id", help="Optional calibrated Wuji SDK user ID")
    parser.add_argument("--tracker-to-wrist-xyz-m", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--tracker-to-wrist-rpy-deg", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("ROLL", "PITCH", "YAW"))
    parser.add_argument("--tracker-zero-frames", type=int, default=20)
    parser.add_argument("--glove-startup-frames", type=int, default=10)
    parser.add_argument("--input-timeout", type=float, default=0.25, help="Glove/Tracker freshness limit in seconds")
    parser.add_argument("--task-name", default="default_task", help="Task name for server-side file naming")
    args = parser.parse_args()
    if args.input_device == "glove" and (not args.glove_sn or not args.tracker_serial):
        parser.error("--devise glove requires --glove-sn and --tracker-serial")
    if args.input_timeout <= 0 or args.tracker_zero_frames < 2 or args.glove_startup_frames < 1:
        parser.error("Glove timeouts and startup frame counts must be positive")
    if not np.isfinite([*args.tracker_to_wrist_xyz_m, *args.tracker_to_wrist_rpy_deg]).all():
        parser.error("Tracker-to-wrist transform must be finite")
    if args.hand_side == "left" and args.input_device != "glove":
        parser.error("Left-hand input currently requires --devise glove")
    if args.tracker_source == "udp" and not args.tracker_source_ip:
        parser.error("UDP tracker requires --tracker-source-ip")
    return args


def wait_for_shm_info(img_sub, addr_desc: str, timeout_s: float = 30.0):
    """Wait for the server to send image metadata (Handshake)."""
    poller = zmq.Poller()
    poller.register(img_sub, zmq.POLLIN)
    start = time.time()

    print(f"[Teleop] Connecting to image stream ({addr_desc})... Waiting for handshake...")

    while True:
        if (time.time() - start) > timeout_s:
            raise TimeoutError("Timeout waiting for server handshake. Is the server running?")

        socks = dict(poller.poll(timeout=100))
        if img_sub in socks and socks[img_sub] == zmq.POLLIN:
            try:
                frames = img_sub.recv_multipart()
                meta = json.loads(frames[0].decode('utf-8'))

                shm_name = meta.get("shm")
                shape = meta.get("shape")
                net_mode = bool(meta.get("net", False))

                if shape:
                    return (
                        shm_name,
                        tuple(shape),
                        net_mode,
                        meta.get("camera_mode", "both"),
                        meta.get("preview_camera", "ZED_M"),
                    )
            except Exception as e:
                print(f"[Teleop] Handshake parse error: {e}, retrying...")
                continue


def keyboard_listener(cmd_queue: Queue):
    """Thread to handle keyboard input."""
    print("\n" + "=" * 40)
    print(" [KEYBOARD CONTROLS]")
    print("  T : Toggle Teleoperation (ON/OFF)")
    print("  R : Toggle Recording (Server-side)")
    print("  Q : Quit")
    print("=" * 40 + "\n")

    while True:
        try:
            ch = input().strip().lower()
            if ch == 't':
                cmd_queue.put("toggle_teleop")
            elif ch == 'r':
                cmd_queue.put("toggle_record")
            elif ch == 'q':
                cmd_queue.put("quit")
                return  # Do not hold stdin while the interpreter shuts down.
        except EOFError:
            break


def is_valid_data(wrist_mat, hand_mat):
    """Filter out invalid or default/zero poses from Vuer."""
    if wrist_mat is None or hand_mat is None:
        return False

    hand_flat = hand_mat.flatten()
    # Check for Vuer's specific invalid value marker (often 1.0 or huge floats)
    if abs(hand_flat[0]) > 0.9:
        return False
        # Check for default flat initialization
    if np.any(np.isclose(hand_flat, 0.5)) or np.any(np.isclose(hand_flat, -0.5)):
        return False

    return True
def main():
    args = parse_args()

    # ================= 1. Initialize Retargeting =================
    right_retargeter = None
    if args.input_device == "vr":
        print(f"[Teleop] Initializing Retargeting (Config: {args.retarget_config})...")
        from utils.wuji_retargeting import WujiDexPilotRetargeter
        right_retargeter = WujiDexPilotRetargeter(args.retarget_config)
        print(f"[Teleop] Wuji joint order: {JOINT_NAMES}")
        print(f"[Teleop] Wuji fingertip scales: {right_retargeter.fingertip_scale_summary}")
        print(f"[Teleop] Wuji internal weights: {right_retargeter.internal_constraint_summary}")

    # ================= 2. Network Setup (ZMQ) =================
    ctx = zmq.Context.instance()

    # CMD Socket: PUSH commands to server
    cmd_sock = ctx.socket(zmq.PUSH)
    cmd_sock.setsockopt(zmq.LINGER, 0)
    cmd_sock.setsockopt(zmq.SNDHWM, 1)

    # IMG Socket: SUBscribe to metadata/images from server
    img_sock = ctx.socket(zmq.SUB)
    img_sock.setsockopt(zmq.LINGER, 0)
    img_sock.setsockopt(zmq.SUBSCRIBE, b"")

    print(f"[Teleop] ZMQ Connecting: CMD->{args.cmd_addr}, IMG<-{args.img_addr}")
    cmd_sock.connect(args.cmd_addr)
    img_sock.connect(args.img_addr)

    # ================= 3. Handshake & Initialization =================
    try:
        shm_name, img_shape, net_mode, server_camera_mode, preview_camera = (
            wait_for_shm_info(img_sock, args.img_addr)
        )
        if server_camera_mode != args.camera_mode:
            raise RuntimeError(
                f"camera mode mismatch: client={args.camera_mode}, "
                f"server={server_camera_mode}"
            )
        mode_str = "[NET MODE]" if net_mode else "[SHM MODE]"
        print(
            f"[Teleop] Handshake Success! Mode: {mode_str}, SHM: {shm_name}, "
            f"Cameras: {server_camera_mode}, Preview: {preview_camera}"
        )
    except Exception as e:
        print(f"[Fatal] Initialization failed: {e}")
        return

    if args.input_device == "vr":
        teleoperator = VuerTeleop(shm_name=shm_name, img_shape_override=img_shape)
    else:
        from utils.glove_tracker_input import GloveTrackerInput

        try:
            teleoperator = GloveTrackerInput(
                glove_serial=args.glove_sn,
                hand_side=args.hand_side,
                tracker_source=args.tracker_source,
                tracker_bind=args.tracker_bind,
                tracker_port=args.tracker_port,
                tracker_source_ip=args.tracker_source_ip,
                tracker_serial=args.tracker_serial,
                user_id=args.glove_user_id,
                tracker_to_wrist_xyz_m=args.tracker_to_wrist_xyz_m,
                tracker_to_wrist_rpy_deg=args.tracker_to_wrist_rpy_deg,
                tracker_zero_frames=args.tracker_zero_frames,
                startup_valid_frames=args.glove_startup_frames,
                tracking_timeout_s=args.input_timeout,
            )
        except Exception as exc:
            print(f"[Fatal] Glove/Tracker initialization failed: {exc}")
            cmd_sock.close()
            img_sock.close()
            return

    # Start Keyboard Thread
    kb_queue = Queue()
    kb_thread = threading.Thread(target=keyboard_listener, args=(kb_queue,), daemon=True)
    kb_thread.start()

    # ================= 4. Main Control Loop =================

    # State flags
    teleop_requested = False
    teleop_active = False
    recording_active = False
    running = True

    # Timing variables
    target_dt = 1.0 / args.teleop_hz if args.teleop_hz > 0 else 0.0
    last_send_time = 0.0
    send_counter = 0
    last_stat_time = time.time()
    perf_stats = {}

    def record_perf(name, elapsed_s):
        elapsed_ms = elapsed_s * 1000.0
        stat = perf_stats.setdefault(
            name, {"total_ms": 0.0, "max_ms": 0.0, "count": 0}
        )
        stat["total_ms"] += elapsed_ms
        stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
        stat["count"] += 1

    # ZMQ Poller for non-blocking reads
    poller = zmq.Poller()
    poller.register(img_sock, zmq.POLLIN)

    print(f"[Teleop] System Ready. Current Task Name: [{args.task_name}]")
    print("[Teleop] Press 'T' to start control, 'R' to toggle server recording.")

    try:
        while running:
            # --- A. Process Keyboard Input ---
            try:
                while not kb_queue.empty():
                    evt = kb_queue.get_nowait()

                    if evt == "toggle_teleop":
                        if not teleop_requested:
                            print(f"[Teleop] >>> Request ON. Waiting for valid {args.input_device} data...")
                            teleop_requested = True
                        else:
                            print("[Teleop] >>> Request OFF.")
                            teleop_requested = False
                            teleop_active = False
                            # Send stop command immediately
                            cmd_sock.send_json({"type": "control", "action": "teleop_off"})
                            if args.input_device == "glove" and recording_active:
                                recording_active = False
                                cmd_sock.send_json({"type": "control", "action": "stop_recording"})

                    elif evt == "toggle_record":
                        if args.input_device == "glove" and not teleop_active:
                            print("[Teleop] Recording requires active, fresh Glove/Tracker input.")
                            continue
                        recording_active = not recording_active
                        status = "ON" if recording_active else "OFF"
                        print(f"[Teleop] >>> Toggle Server Recording: {status}")

                        # Send recording command to server
                        cmd_payload = {
                            "type": "control",
                            "action": "start_recording" if recording_active else "stop_recording",
                            "task_name": args.task_name,
                            "input_device": args.input_device,
                        }
                        cmd_sock.send_json(cmd_payload)

                    elif evt == "quit":
                        running = False
                        cmd_sock.send_json({"type": "control", "action": "quit"})

            except Empty:
                pass

            # --- B. Fetch input ---
            perf_started = time.perf_counter()
            if args.input_device == "vr":
                _, _, right_wrist_mat, _, right_hand_mat = teleoperator.step()
                hand_qpos = None
                input_valid = is_valid_data(right_wrist_mat, right_hand_mat)
            else:
                sample = teleoperator.step()
                right_wrist_mat, hand_qpos = sample if sample is not None else (None, None)
                right_hand_mat = None
                input_valid = sample is not None
            record_perf("input", time.perf_counter() - perf_started)

            now = time.time()

            # --- C. Activation Logic ---
            if teleop_requested and not teleop_active:
                if input_valid:
                    print("[Teleop] >>> Valid hand data detected. Teleop ACTIVE!")
                    teleop_active = True
                    # Notify server to compute anchor
                    cmd_sock.send_json({"type": "control", "action": "teleop_on", "hand_side": args.hand_side})
                else:
                    # Log waiting status every second
                    if time.time() - last_stat_time > 1.0:
                        print(f"[Teleop] Waiting for {args.input_device} input...")

            if args.input_device == "glove" and teleop_active and not input_valid:
                print("[Teleop] Glove/Tracker tracking stale; teleoperation disabled.")
                teleop_active = False
                teleop_requested = False
                cmd_sock.send_json({"type": "control", "action": "teleop_off"})
                if recording_active:
                    recording_active = False
                    cmd_sock.send_json({"type": "control", "action": "stop_recording"})

            # --- D. Compute & Send Commands ---
            if teleop_active and (now - last_send_time >= target_dt):
                if input_valid:
                    try:
                        if args.input_device == "vr":
                            r_hand_mat_calc = right_hand_mat.astype(np.float64).reshape(25, 3)
                            perf_started = time.perf_counter()
                            hand_qpos = right_retargeter.retarget(r_hand_mat_calc)
                            record_perf("retarget", time.perf_counter() - perf_started)
                        hand_qpos_list = hand_qpos.tolist()
                        wrist_flat = right_wrist_mat.astype(float).flatten().tolist()

                        # 2. Construct Payload
                        cmd = {
                            "type": "cmd",
                            "wrist_mat": wrist_flat,
                            "right_wrist_mat": wrist_flat,
                            "hand_side": args.hand_side,
                            "hand_qpos": hand_qpos_list
                        }

                        # 3. Send
                        perf_started = time.perf_counter()
                        cmd_sock.send_json(cmd, flags=zmq.NOBLOCK)
                        record_perf("zmq_send", time.perf_counter() - perf_started)
                        send_counter += 1

                    except zmq.Again:
                        pass  # Network busy, skip frame
                    except Exception as e:
                        print(f"[Error] Retargeting/Send failed: {e}")

                    last_send_time = now

            # --- E. Handle Image Stream (Keep Network Alive) ---
            try:
                while True:
                    socks = dict(poller.poll(0))
                    if img_sock not in socks:
                        break

                    perf_started = time.perf_counter()
                    frames = img_sock.recv_multipart()

                    if args.input_device == "vr" and net_mode and len(frames) >= 2:
                        meta = json.loads(frames[0].decode('utf-8'))
                        img_fmt = meta.get("format", "raw")
                        img_buf = np.frombuffer(frames[1], dtype=np.uint8)

                        if img_fmt == "jpg":
                            decoded_img = cv2.imdecode(img_buf, cv2.IMREAD_COLOR)

                            if decoded_img is not None:
                                decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

                                if decoded_img.size == teleoperator.img_array.size:
                                    teleoperator.img_array[:] = decoded_img.reshape(teleoperator.img_shape)
                        else:
                            if img_buf.size == teleoperator.img_array.size:
                                teleoperator.img_array[:] = img_buf.reshape(teleoperator.img_shape)
                    record_perf("image", time.perf_counter() - perf_started)
            except Exception:
                pass

            # --- F. Statistics Log ---
            if time.time() - last_stat_time >= 1.0:
                if teleop_active:
                    rec_str = "[REC]" if recording_active else ""
                    print(f"[Teleop] Send Rate: {send_counter} Hz {rec_str}")
                labels = {
                    "input": args.input_device,
                    "retarget": "Retarget",
                    "zmq_send": "ZmqSend",
                    "image": "Image",
                }
                details = []
                for key, label in labels.items():
                    stat = perf_stats.get(key)
                    if stat and stat["count"]:
                        details.append(
                            f"{label}:{stat['total_ms'] / stat['count']:.1f}/"
                            f"{stat['max_ms']:.1f}#{stat['count']}"
                        )
                # if details:
                #     print(
                #         "[ClientDiag] avg/max(ms)#count | " + " ".join(details),
                #         flush=True,
                #     )
                perf_stats.clear()
                send_counter = 0
                last_stat_time = time.time()

            # Sleep to prevent CPU burning
            time.sleep(0.0005)

    except KeyboardInterrupt:
        print("\n[Teleop] Interrupted by user.")
    finally:
        try:
            cmd_sock.send_json({"type": "control", "action": "teleop_off"}, flags=zmq.NOBLOCK)
        except zmq.ZMQError:
            pass
        teleoperator.close()
        cmd_sock.close(linger=0)
        img_sock.close(linger=0)
        ctx.term()
        print("[Teleop] Shutdown complete.")


if __name__ == "__main__":
    main()
