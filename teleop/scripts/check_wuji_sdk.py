#!/usr/bin/env python3
"""Read-only preflight check for the first-generation right Wuji Hand."""

import argparse
from pathlib import Path
import sys

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
        HOME_Q,
        JOINT_LIMITS,
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
    parser.add_argument(
        "--device",
        default=None,
        help="USB serial number; omitted selects the connected right hand by side",
    )
    return parser.parse_args()


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
    version = getattr(wujihandpy, "__version__", "unknown")
    print(f"[SDK] wujihandpy {version}: {wujihandpy.__file__}")
    print("[SAFE] This check never enables the motors or sends target positions.")

    hand = None
    try:
        if args.device:
            hand = wujihandpy.Hand(serial_number=args.device)
        else:
            hand = wujihandpy.Hand(side="right")

        handedness = int(hand.read_handedness())
        if handedness != 0:
            raise RuntimeError(
                f"device handedness={handedness}; this pipeline requires right=0"
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

        safe_lower = np.maximum(JOINT_LIMITS[:, 0], lower) + HAND_LIMIT_MARGIN
        safe_upper = np.minimum(JOINT_LIMITS[:, 1], upper) - HAND_LIMIT_MARGIN
        if np.any(safe_lower >= safe_upper):
            raise RuntimeError("URDF/hardware limit intersection is empty")
        if np.any(HOME_Q < safe_lower) or np.any(HOME_Q > safe_upper):
            raise RuntimeError("configured HOME pose is outside the safe limit intersection")

        print(f"[DEVICE] handedness=right, voltage={voltage:.2f} V, firmware=0x{firmware:08x}")
        print(f"[JOINTS] {HAND_DOF} feedback values and calibrated limits are valid")
        if np.any(errors != 0):
            bad = [
                f"{JOINT_NAMES[i]}=0x{int(errors[i]):08x}"
                for i in np.flatnonzero(errors)
            ]
            raise RuntimeError("joint error codes present: " + ", ".join(bad))

        for name, q, lo, hi in zip(JOINT_NAMES, actual, safe_lower, safe_upper):
            print(f"  {name:<15} q={q:+.4f}  safe=[{lo:+.4f}, {hi:+.4f}] rad")
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
