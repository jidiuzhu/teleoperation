#!/usr/bin/env python3
"""Offline ABI/FK/IK check, or read-only MARVIN 100341 feedback probe."""
import argparse
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.marvin_kinematics import MarvinKinematics, VENDOR_ROOT, DEFAULT_CONFIG
from utils.marvin_controller import MarvinController, _NativeSDK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-ip', help='Omit for offline checks (no robot connection)')
    parser.add_argument('--side', choices=('A','B'), default='A')
    parser.add_argument('--sdk-root', type=Path, default=VENDOR_ROOT)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--duration', type=float, default=10)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error('--duration must be positive')
    kine = MarvinKinematics(args.config, args.side, args.sdk_root)
    _NativeSDK(args.sdk_root, args.side)  # Load and check ABI; does not connect.
    print('Native libraries and ctypes layouts: OK')
    if not args.arm_ip:
        # Sample is for offline calculation only, never a motion target.
        q = np.deg2rad([21.8,-41,-4.74,-63.67,10.15,14.72,7.68])
        pos, quat = kine.compute_fk(q)
        solved = kine.solve_ik(q,pos,quat,initial_q=q)
        if not kine.last_solve_info['success']:
            raise RuntimeError(kine.last_solve_info)
        print('Offline FK position (m):',pos)
        print('FK -> IK max joint error (deg):',np.max(np.abs(np.rad2deg(solved-q))))
        return
    arm = MarvinController(args.arm_ip, side=args.side, sdk_root=args.sdk_root,
                            config_path=args.config, enable_motion=False)
    try:
        print(f'Controller version: {arm.version}; READ ONLY')
        end = time.monotonic()+args.duration
        while time.monotonic() < end:
            ok, q = arm.get_joint_state()
            if not ok:
                raise RuntimeError('Missing/invalid MARVIN feedback')
            pos, _ = kine.compute_fk(np.deg2rad(q))
            print(f'q_deg={np.round(q,3)} flange_m={np.round(pos,4)}',flush=True)
            time.sleep(0.2)
    finally:
        arm.disconnect()


if __name__ == '__main__':
    main()
