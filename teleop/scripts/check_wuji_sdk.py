#!/usr/bin/env python3
"""Read-only preflight check for the first-generation Wuji Hand."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np

TELEOP_ROOT = Path(__file__).resolve().parents[1]
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

try:
    import wujihandpy
except ImportError as exc:
    project_root = Path(__file__).resolve().parents[2]
    source_checkout = project_root / "wujihandpy-main"
    print("[FAIL] wujihandpy is not installed in this Python environment.")
    if source_checkout.is_dir():
        print(f"       Found SDK source at {source_checkout}.")
        print("       The source has a native _core extension and is not directly importable.")
    print("       Install the wheel: python -m pip install wujihandpy==1.8.0")
    raise SystemExit(2) from exc

try:
    from utils.wuji_hand_spec import (
        HAND_DOF,
        hand_spec,
        JOINT_NAMES,
    )
except ImportError as exc:
    print(f"[FAIL] Cannot load the Wuji model specification: {exc}")
    raise SystemExit(2) from exc


HAND_LIMIT_MARGIN = 0.01


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read Wuji Hand identity, feedback and limits without enabling motors"
    )
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument(
        "--device",
        default=None,
        help="USB serial number; omitted selects the connected hand matching --side",
    )
    parser.add_argument('--samples', type=int, default=1,
                        help='Number of read-only feedback samples (1..200)')
    args = parser.parse_args()
    if not 1 <= args.samples <= 200:
        parser.error('--samples must be 1..200')
    return args


def sdk_vector(values, label):
    array = np.asarray(values)
    if array.shape != (5, 4):
        raise ValueError(f"{label} must have SDK shape (5, 4), got {array.shape}")
    flat = array.astype(np.float64, copy=False).reshape(HAND_DOF)
    if not np.all(np.isfinite(flat)):
        raise ValueError(f"{label} contains non-finite values")
    return flat


def main():
    args = parse_args()
    _, _, model_limits = hand_spec(args.side)
    version = getattr(wujihandpy, "__version__", "unknown")
    print(f"[SDK] wujihandpy {version}: {wujihandpy.__file__}")
    print("[SAFE] This check never enables the motors or sends target positions.")

    hand = None
    try:
        if args.device:
            hand = wujihandpy.Hand(serial_number=args.device)
        else:
            hand = wujihandpy.Hand(side=args.side)

        handedness = int(hand.read_handedness())
        if handedness != (0 if args.side == "right" else 1):
            raise RuntimeError(
                f"device handedness={handedness}; expected {args.side}"
            )

        actual = sdk_vector(hand.read_joint_actual_position(), "actual position")
        lower = sdk_vector(hand.read_joint_lower_limit(), "hardware lower limits")
        upper = sdk_vector(hand.read_joint_upper_limit(), "hardware upper limits")
        errors = np.asarray(hand.read_joint_error_code()).reshape(HAND_DOF)
        voltage = float(hand.read_input_voltage())
        firmware = int(hand.read_full_system_firmware_version())

        if np.any(lower >= upper):
            raise RuntimeError("hardware returned lower limits greater than upper limits")
        if np.any(actual < lower) or np.any(actual > upper):
            raise RuntimeError("actual joint feedback is outside hardware limits")

        safe_lower = np.maximum(model_limits[:, 0], lower) + HAND_LIMIT_MARGIN
        safe_upper = np.minimum(model_limits[:, 1], upper) - HAND_LIMIT_MARGIN
        if np.any(safe_lower >= safe_upper):
            raise RuntimeError("URDF/hardware limit intersection is empty")

        print(f"[DEVICE] handedness={args.side}, voltage={voltage:.2f} V, firmware=0x{firmware:08x}")
        print(f"[IDENTITY] requested_usb_serial={args.device or 'auto-by-side'} product_sn={hand.get_product_sn()}")
        print(f"[JOINTS] {HAND_DOF} feedback values and calibrated limits are valid")
        if np.any(errors != 0):
            bad = [
                f"{JOINT_NAMES[i]}=0x{int(errors[i]):08x}"
                for i in np.flatnonzero(errors)
            ]
            raise RuntimeError("joint error codes present: " + ", ".join(bad))

        for name, q, lo, hi in zip(JOINT_NAMES, actual, safe_lower, safe_upper):
            print(f"  {name:<15} q={q:+.4f}  safe=[{lo:+.4f}, {hi:+.4f}] rad")
        if args.samples > 1:
            samples = [actual.copy()]
            print(f"[STABILITY] Reading {args.samples} samples without enable or realtime controller", flush=True)
            for _ in range(args.samples-1):
                time.sleep(0.1)
                sample = sdk_vector(hand.read_joint_actual_position(), "stability feedback")
                if np.any(sample < lower) or np.any(sample > upper):
                    raise RuntimeError("stability feedback outside hardware limits")
                errors = np.asarray(hand.read_joint_error_code()).reshape(HAND_DOF)
                if np.any(errors != 0):
                    raise RuntimeError(f"joint errors during sampling: {errors.tolist()}")
                samples.append(sample)
            values = np.asarray(samples)
            spans = np.ptp(values, axis=0)
            for i in np.argsort(spans)[::-1][:5]:
                print(f"[STABILITY] {JOINT_NAMES[i]} min={values[:,i].min():+.6f} "
                      f"max={values[:,i].max():+.6f} span={spans[i]:.6f} rad", flush=True)
            print("[STABILITY] Static range only; this does not validate enable behavior.")
        print("[PASS] Wuji SDK read-only preflight completed")
        return 0
    except Exception as exc:
        print(f"[FAIL] Wuji SDK preflight failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        # Hand has no public close method. Releasing the last reference invokes
        # the native destructor; no controller was created and motors were never enabled.
        hand = None


if __name__ == "__main__":
    raise SystemExit(main())
