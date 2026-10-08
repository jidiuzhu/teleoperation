#!/usr/bin/env python3
"""VR retargeting test for a standalone right Wuji Hand, without a robot arm."""

import argparse
from multiprocessing import Event, Queue as ProcessQueue, shared_memory
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import time

import numpy as np


TELEOP_ROOT = Path(__file__).resolve().parents[1]
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

from utils.XHand_controller_wuji import WujiHandDriver
from utils.wuji_hand_spec import (
    JOINT_LIMITS,
    JOINT_NAMES,
    validate_joint_vector,
)
from utils.wuji_retargeting import WujiDexPilotRetargeter


HAND_LIMIT_MARGIN = 0.01


def positive_float(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return number


def parse_args():
    parser = argparse.ArgumentParser(
        description="Control a standalone right Wuji Hand from Vuer/DexPilot"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Wuji USB serial number; omitted selects the connected right hand",
    )
    parser.add_argument(
        "--retarget-config",
        type=Path,
        default=TELEOP_ROOT / "teleop" / "wuji.yml",
        help="Wuji DexPilot configuration",
    )
    parser.add_argument("--control-hz", type=positive_float, default=30.0)
    parser.add_argument(
        "--max-joint-speed",
        type=positive_float,
        default=0.55,
        help="Per-joint command speed limit in rad/s",
    )
    parser.add_argument(
        "--max-tracking-error",
        type=positive_float,
        default=0.45,
        help="Maximum sustained command/feedback error in radians",
    )
    parser.add_argument(
        "--tracking-error-timeout",
        type=positive_float,
        default=0.75,
        help="Seconds a tracking error may remain over its limit",
    )
    parser.add_argument(
        "--vr-timeout",
        type=positive_float,
        default=0.5,
        help="Disable the hand when Vuer right-hand updates are this old",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument(
        "--ngrok",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the existing Vuer non-certificate launch mode",
    )
    parser.add_argument("--cert-file", default="./cert.pem")
    parser.add_argument("--key-file", default="./key.pem")
    args = parser.parse_args()
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive")
    return args


def valid_raw_right_hand(wrist_matrix, landmarks):
    wrist = np.asarray(wrist_matrix, dtype=np.float64)
    points = np.asarray(landmarks, dtype=np.float64)
    if wrist.shape != (4, 4) or points.shape != (25, 3):
        return False
    if not np.all(np.isfinite(wrist)) or not np.all(np.isfinite(points)):
        return False
    if np.allclose(wrist, 0.0) or np.allclose(points, 0.0):
        return False

    rotation = wrist[:3, :3]
    determinant = np.linalg.det(rotation)
    if not np.isfinite(determinant) or abs(determinant) < 0.5:
        return False

    hand_extent = np.max(np.linalg.norm(points - points[0], axis=1))
    return 0.03 <= hand_extent <= 0.50


class WujiHandControlSession:
    def __init__(
        self,
        *,
        device,
        max_joint_speed,
        max_tracking_error,
        tracking_error_timeout,
        backend_spec="wujihandpy",
    ):
        self._driver = None
        self.max_joint_speed = float(max_joint_speed)
        self.max_tracking_error = float(max_tracking_error)
        self.tracking_error_timeout = float(tracking_error_timeout)
        self._tracking_error_since = None

        try:
            self._driver = WujiHandDriver(
                device=device, backend_spec=backend_spec
            )
            hardware_limits = self._driver.get_joint_limits()
            self.safe_lower = (
                np.maximum(JOINT_LIMITS[:, 0], hardware_limits[:, 0])
                + HAND_LIMIT_MARGIN
            )
            self.safe_upper = (
                np.minimum(JOINT_LIMITS[:, 1], hardware_limits[:, 1])
                - HAND_LIMIT_MARGIN
            )
            if np.any(self.safe_lower >= self.safe_upper):
                raise RuntimeError("URDF/hardware safe limit intersection is empty")

            current = self._driver.get_current_joints()
            if np.any(current < hardware_limits[:, 0]) or np.any(
                current > hardware_limits[:, 1]
            ):
                raise RuntimeError("Initial Wuji feedback is outside hardware limits")
            outside_safe = np.flatnonzero(
                (current < self.safe_lower) | (current > self.safe_upper)
            )
            if outside_safe.size:
                names = [JOINT_NAMES[index] for index in outside_safe]
                raise RuntimeError(
                    "Initial Wuji feedback is outside the URDF/hardware safe "
                    f"intersection at joints: {names}"
                )
            self.command = current.copy()
            self.feedback = current.copy()
            self._driver.send_joints(self.command)
        except Exception:
            self.close()
            raise

    def step(self, raw_target, dt, now=None):
        target = validate_joint_vector(raw_target, label="Wuji VR target")
        safe_target = np.clip(target, self.safe_lower, self.safe_upper)
        clipped_count = int(np.count_nonzero(np.abs(safe_target - target) > 1e-9))

        bounded_dt = min(max(float(dt), 0.0), 0.1)
        max_step = self.max_joint_speed * bounded_dt
        delta = np.clip(safe_target - self.command, -max_step, max_step)
        next_command = np.clip(
            self.command + delta, self.safe_lower, self.safe_upper
        )

        self._driver.send_joints(next_command)
        feedback = self._driver.get_current_joints()
        self.command = next_command
        self.feedback = feedback

        errors = np.abs(feedback - next_command)
        max_error_index = int(np.argmax(errors))
        max_error = float(errors[max_error_index])
        check_time = time.monotonic() if now is None else float(now)
        if max_error > self.max_tracking_error:
            if self._tracking_error_since is None:
                self._tracking_error_since = check_time
            elif check_time - self._tracking_error_since >= self.tracking_error_timeout:
                raise RuntimeError(
                    "Wuji tracking error remained above the safety limit: "
                    f"{JOINT_NAMES[max_error_index]}={max_error:.3f} rad"
                )
        else:
            self._tracking_error_since = None

        return {
            "clipped_count": clipped_count,
            "max_error": max_error,
            "max_error_joint": JOINT_NAMES[max_error_index],
        }

    def close(self):
        driver = self._driver
        self._driver = None
        if driver is not None:
            driver.close()


def keyboard_worker(events):
    while True:
        try:
            key = input().strip().lower()
        except EOFError:
            return
        if key == "t":
            events.put("toggle")
        elif key == "q":
            events.put("quit")
            return


def close_hand_session(session, reason):
    if session is not None:
        session.close()
        print(f"[SAFE] Wuji Hand disabled: {reason}")
    return None


def main():
    args = parse_args()
    print(f"[Init] Loading Wuji DexPilot config: {args.retarget_config}")
    try:
        retargeter = WujiDexPilotRetargeter(args.retarget_config)
    except Exception as exc:
        print(f"[Fatal] Retargeting initialization failed: {exc}")
        return 2
    print(f"[Init] Wuji fingertip scales: {retargeter.fingertip_scale_summary}")
    print(
        "[Init] Wuji internal weights (wrist-to-joint/joint-to-tip): "
        f"{retargeter.internal_constraint_summary}"
    )

    original_argv = sys.argv[:]
    try:
        # Vuer uses params_proto and inspects sys.argv while importing. Keep
        # its CLI separate from this standalone controller's argparse CLI.
        sys.argv = [sys.argv[0]]
        from teleop.Preprocessor_wuji import VuerPreprocessorWuji
        from teleop.TeleVision_wuji import OpenTeleVisionWuji
    finally:
        sys.argv = original_argv

    image_shape = (args.height, args.width * 2, 3)
    image_bytes = int(np.prod(image_shape) * np.dtype(np.uint8).itemsize)
    shm = shared_memory.SharedMemory(create=True, size=image_bytes)
    image = np.ndarray(image_shape, dtype=np.uint8, buffer=shm.buf)
    image.fill(0)

    television = None
    hand_session = None
    try:
        television = OpenTeleVisionWuji(
            (args.height, args.width),
            shm.name,
            ProcessQueue(),
            Event(),
            cert_file=args.cert_file,
            key_file=args.key_file,
            ngrok=args.ngrok,
        )
        preprocessor = VuerPreprocessorWuji()

        keyboard_events = Queue()
        threading.Thread(
            target=keyboard_worker, args=(keyboard_events,), daemon=True
        ).start()

        print("[Ready] Open the Vuer session and expose your right hand to tracking.")
        print("[Ready] Press T then Enter to enable/disable hand control; Q then Enter to quit.")
        print("[SAFE] The hand remains disabled until T is pressed and valid VR data is current.")

        requested = False
        running = True
        last_step_time = time.monotonic()
        next_step_time = last_step_time
        last_status_time = 0.0
        step_count = 0
        status_clipped_count = 0
        status_max_error = 0.0
        status_max_error_joint = JOINT_NAMES[0]

        while running:
            try:
                while True:
                    event = keyboard_events.get_nowait()
                    if event == "quit":
                        running = False
                        break
                    if event == "toggle":
                        if requested or hand_session is not None:
                            requested = False
                            hand_session = close_hand_session(
                                hand_session, "operator requested OFF"
                            )
                        else:
                            requested = True
                            print("[Control] ON requested; waiting for a valid fresh right hand.")
            except Empty:
                pass
            if not running:
                break

            process = getattr(television, "process", None)
            if process is not None and not process.is_alive():
                raise RuntimeError("Vuer process exited unexpectedly")

            now = time.monotonic()
            if now < next_step_time:
                time.sleep(min(0.002, next_step_time - now))
                continue
            next_step_time = now + 1.0 / args.control_hz

            raw_wrist = television.right_hand
            raw_landmarks = television.right_landmarks
            vr_age = television.right_hand_update_age
            valid_vr = (
                vr_age <= args.vr_timeout
                and valid_raw_right_hand(raw_wrist, raw_landmarks)
            )

            if not valid_vr:
                if hand_session is not None:
                    hand_session = close_hand_session(
                        hand_session,
                        f"VR tracking stale/invalid (age={vr_age:.3f}s)",
                    )
                    requested = False
                if now - last_status_time >= 1.0:
                    last_status_time = now
                last_step_time = now
                continue

            try:
                _, _, _, _, right_hand = preprocessor.process(television, "wuji")
                target = retargeter.retarget(right_hand)
            except Exception as exc:
                hand_session = close_hand_session(
                    hand_session, f"retargeting failed: {exc}"
                )
                requested = False
                print(f"[Error] Retargeting failed: {exc}")
                last_step_time = now
                continue

            if requested and hand_session is None:
                try:
                    hand_session = WujiHandControlSession(
                        device=args.device,
                        max_joint_speed=args.max_joint_speed,
                        max_tracking_error=args.max_tracking_error,
                        tracking_error_timeout=args.tracking_error_timeout,
                    )
                    last_step_time = now
                    print(
                        "[Control] ACTIVE. Starting from measured joint feedback; "
                        f"speed limit={args.max_joint_speed:.3f} rad/s."
                    )
                except Exception as exc:
                    requested = False
                    print(f"[Error] Cannot enable Wuji Hand: {exc}")
                    last_step_time = now
                    continue

            if hand_session is not None:
                try:
                    status = hand_session.step(target, now - last_step_time, now=now)
                    step_count += 1
                    status_clipped_count = max(
                        status_clipped_count, status["clipped_count"]
                    )
                    if status["max_error"] >= status_max_error:
                        status_max_error = status["max_error"]
                        status_max_error_joint = status["max_error_joint"]
                except Exception as exc:
                    hand_session = close_hand_session(
                        hand_session, f"control failure: {exc}"
                    )
                    requested = False
                    print(f"[Error] Wuji control stopped: {exc}")

            last_step_time = now
            if now - last_status_time >= 1.0:
                state = "ACTIVE" if hand_session is not None else "READY"
                print(
                    f"[{state}] rate={step_count} Hz, vr_age={vr_age:.3f}s, "
                    f"max_error={status_max_error:.3f} rad "
                    f"({status_max_error_joint}), clipped_joints={status_clipped_count}"
                )
                step_count = 0
                status_clipped_count = 0
                status_max_error = 0.0
                last_status_time = now

        return 0
    except KeyboardInterrupt:
        print("\n[Exit] Interrupted by user.")
        return 0
    except Exception as exc:
        print(f"[Fatal] Standalone Wuji teleoperation stopped: {exc}")
        return 1
    finally:
        hand_session = close_hand_session(hand_session, "program exit")
        if television is not None:
            television.close()
        shm.close()
        shm.unlink()
        print("[Exit] Vuer resources released.")


if __name__ == "__main__":
    raise SystemExit(main())
