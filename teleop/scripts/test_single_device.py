#!/usr/bin/env python3
"""Read-only plan by default; explicitly authorized arm OR hand joint excursion."""
import argparse
from pathlib import Path
import signal
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.limited_joint_test import plan_single_joint, run_single_joint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='device',required=True)
    for name, count, default, max_delta in [('arm',7,1.0,2.0),('hand',20,.03,.05)]:
        p = sub.add_parser(name)
        p.add_argument('--joint',type=int,required=True,help=f'1..{count}; hand: thumb-to-little, four joints per finger')
        p.add_argument('--delta',type=float,default=default,help='Degrees for arm; radians for hand')
        p.add_argument('--duration',type=float,default=10,help='8..20 seconds for out-and-back')
        p.add_argument('--enable-motion',action='store_true')
        p.add_argument('--confirmation',choices=('MOVE_ARM' if name=='arm' else 'MOVE_HAND',))
        if name=='arm':
            p.add_argument('--arm-ip',required=True)
            p.add_argument('--side',choices=('A','B'),default='A')
        else:
            p.add_argument('--hand-device',required=True,help='USB serial, not glove SN')
            p.add_argument('--side',choices=('left','right'),default='left')
    args=parser.parse_args(argv)
    limit=2.0 if args.device=='arm' else .05
    count=7 if args.device=='arm' else 20
    if not 1 <= args.joint <= count:
        parser.error('Joint index out of range')
    if not np.isfinite(args.delta) or not 0 < abs(args.delta) <= limit:
        parser.error(f'Absolute delta must be positive and <= {limit}')
    if not np.isfinite(args.duration) or not 8 <= args.duration <= 20:
        parser.error('Duration must be 8..20 seconds')
    if args.enable_motion != (args.confirmation is not None):
        parser.error('Motion requires BOTH --enable-motion and matching --confirmation')
    return args


class ArmTestDevice:
    def __init__(self,args):
        from utils.marvin_controller import MarvinController
        self.backend=MarvinController(args.arm_ip,side=args.side,
            enable_motion=args.enable_motion,velocity_ratio=1,acceleration_ratio=1,
            max_joint_speed_deg_s=2,command_timeout=.3)
    def read(self):
        ok,q=self.backend.get_joint_state()
        if not ok: raise RuntimeError('No valid arm feedback')
        return np.array(q)
    def validate(self,q):return self.backend.model.validate_degrees(q,margin=.5)
    def enable(self):self.backend.enable()
    def send(self,q):self.backend.move_j_stream(q)
    def stop(self):self.backend.stop()
    def close(self):self.backend.disconnect()


class HandTestDevice:
    def __init__(self,args):
        from utils.XHand_controller_wuji import WujihandPyBackend
        from utils.wuji_hand_spec import hand_spec
        self.backend=WujihandPyBackend(device=args.hand_device,side=args.side,
            enable_motion=args.enable_motion,defer_enable=True,effort_limit=.3)
        try:
            hw=self.backend.get_joint_limits();model=hand_spec(args.side)[2]
            self.lo=np.maximum(hw[:,0],model[:,0])+.01
            self.hi=np.minimum(hw[:,1],model[:,1])-.01
            if np.any(self.lo>=self.hi):raise ValueError('Empty hand joint ranges')
        except BaseException:
            self.backend.close();raise
    def read(self):return self.backend.read_checked_positions()
    def validate(self,q):
        q=np.asarray(q,float)
        if q.shape!=(20,) or not np.isfinite(q).all() or np.any(q<self.lo) or np.any(q>self.hi):
            raise ValueError('Hand feedback/target outside conservative limits')
    def enable(self):self.backend.enable()
    def send(self,q):self.backend.send_joint_positions(q)
    def stop(self):self.backend.close()
    def close(self):self.backend.close()


def main():
    args=parse_args()
    def interrupted(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    device=None
    try:
        device=ArmTestDevice(args) if args.device=='arm' else HandTestDevice(args)
        start,target=plan_single_joint(device.read(),args.joint-1,args.delta,device.validate)
        unit='deg' if args.device=='arm' else 'rad'
        print(f'{args.device} {args.side}: joint {args.joint}, {start[args.joint-1]:.5f} -> {target[args.joint-1]:.5f} -> start ({unit})',flush=True)
        print('Other joint targets remain at measured starting angles; their servos also become enabled.',flush=True)
        if not args.enable_motion:
            print('READ-ONLY PLAN: no motors enabled and no targets sent.');return
        print('Motion begins in 5 seconds. Ctrl+C cancels. Clear the workspace and keep physical emergency stop available.',flush=True)
        for _ in range(5):
            actual=device.read();device.validate(actual)
            if np.max(np.abs(actual-start)) > (.2 if args.device=='arm' else .01):
                raise RuntimeError('Starting pose changed during countdown')
            time.sleep(1)
        run_single_joint(device,start,target,device.validate,duration=args.duration,
            tracking_tolerance=1.0 if args.device=='arm' else .04,
            completion_tolerance=.2 if args.device=='arm' else .01,
            feedback_speed_limit=5 if args.device=='arm' else .2)
        print('PASS: measured feedback settled near the starting pose. Device stopped.',flush=True)
    finally:
        if device is not None:device.close()


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Interrupted; stop requested. No automatic return motion.')
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
