#!/usr/bin/env python3
"""One bounded LEFT index-J2 outward adjustment; read-only unless --apply.

Uses the left model to require increasing index/middle tip distance. This is
not collision detection for the glove cover. Never returns to the old pose;
after a short settling check it disables the hand (no active hold guarantee).
"""
import argparse
from pathlib import Path
import signal
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from utils.wuji_fingertip_kinematics import _UrdfTree
from utils.wuji_hand_spec import JOINT_NAMES, hand_spec

INDEX = 5  # zero-based: finger(1).joint(1), index finger J2


def vector(raw):
    q=np.asarray(raw,dtype=float)
    if q.shape!=(5,4) or not np.isfinite(q).all():
        raise ValueError('Expected finite 5x4 feedback')
    return q.reshape(20).copy()


def spacing_plan(q, lower, upper, delta):
    if not np.isfinite(delta) or not -.02 <= delta <= -.005:
        raise ValueError('Only one outward step of -0.005..-0.02 rad is allowed')
    model,path,limits=hand_spec('left')
    lo=np.maximum(lower,limits[:,0])+.01
    hi=np.minimum(upper,limits[:,1])-.01
    if q.shape!=(20,) or not np.isfinite(q).all() or np.any(lo>=hi):
        raise ValueError('Invalid feedback/limits')
    target=q.copy();target[INDEX]+=delta
    tree=_UrdfTree(path)
    distances=[]
    for fraction in np.linspace(0,1,21):
        sample=q+(target-q)*fraction
        if np.any(sample<lo) or np.any(sample>hi):raise ValueError('Plan exceeds hand limits')
        positions=dict(zip(JOINT_NAMES,sample))
        a=tree.transform('finger2_tip_link',positions)[:3,3]
        b=tree.transform('finger3_tip_link',positions)[:3,3]
        distances.append(float(np.linalg.norm(a-b)))
    if np.any(np.diff(distances)<=0):raise ValueError('Model does not predict monotonically increasing tip spacing')
    return target,lo,hi,distances[0],distances[-1]


def read_checked(hand):
    q=vector(hand.read_joint_actual_position(timeout=.15))
    errors=np.asarray(hand.read_joint_error_code(timeout=.15))
    if errors.shape!=(5,4) or np.any(errors!=0):raise RuntimeError('Hand reports joint errors')
    return q


def apply_once(hand,sdk,start,target,lo,hi,*,clock=time.monotonic,sleep=time.sleep):
    controller=None
    touched=False
    try:
        # Failures never restore a previous enable mask or repeat the move.
        touched=True
        hand.write_joint_enabled(False,timeout=.15)
        current=read_checked(hand)
        if np.max(np.abs(current-start))>.01:raise RuntimeError('Pose changed before enable')
        joint=hand.finger(1).joint(1)
        existing=float(joint.read_joint_effort_limit(timeout=.15))
        if not np.isfinite(existing) or existing<=0:raise RuntimeError('Invalid existing effort limit')
        limit=min(existing,.3)
        joint.write_joint_effort_limit(limit,timeout=.15)
        if abs(float(joint.read_joint_effort_limit(timeout=.15))-limit)>.001:
            raise RuntimeError('Selected joint effort limit readback failed')
        controller=hand.realtime_controller(enable_upstream=True,filter=sdk.filter.LowPass(cutoff_freq=5.0))
        controller.set_joint_target_position(start.reshape(5,4))
        # Prime while disabled; only index J2 is subsequently enabled.
        sleep(.1)
        current=read_checked(hand)
        if np.max(np.abs(current-start))>.01:raise RuntimeError('Feedback changed during disabled controller setup')
        mask=np.zeros((5,4),dtype=bool);mask[1,1]=True
        hand.write_joint_enabled(mask,timeout=.15)
        began=previous=clock();previous_target=start.copy();settled=0
        while True:
            now=clock()
            if now-previous>.25:raise TimeoutError('Adjustment loop stalled')
            read_start=clock();actual=read_checked(hand);now=clock()
            if now-read_start>.2:raise TimeoutError('Feedback too slow')
            if np.any(actual<lo) or np.any(actual>hi):raise RuntimeError('Feedback outside limits')
            if np.max(np.abs(actual-previous_target))>.04:raise RuntimeError('Tracking error >0.04 rad')
            other=np.arange(20)!=INDEX
            if np.max(np.abs(actual[other]-start[other]))>.04:raise RuntimeError('Nonselected joint shifted >0.04 rad')
            if actual[INDEX]>start[INDEX]+.005 or actual[INDEX]<target[INDEX]-.01:
                raise RuntimeError('Selected joint moved opposite to plan or beyond endpoint')
            elapsed=now-began
            # Fixed 4s ramp; no catch-up beyond the maximum configured step.
            fraction=min(elapsed/4,1)
            command=start+(target-start)*fraction
            controller.set_joint_target_position(command.reshape(5,4))
            previous_target=command
            if fraction>=1:
                settled=settled+1 if abs(actual[INDEX]-target[INDEX])<.005 else 0
                if settled>=5:
                    if start[INDEX]-actual[INDEX] < abs(target[INDEX]-start[INDEX])*.5:
                        raise RuntimeError('Insufficient observed outward change')
                    return actual
                if elapsed>6:raise TimeoutError('Selected joint did not settle')
            previous=now;sleep(.05)
    finally:
        if touched:
            try:
                hand.write_joint_enabled(False,timeout=.15)
            except Exception as exc:
                raise RuntimeError('Could not confirm disable; use physical power stop') from exc
            finally:
                if controller is not None:controller.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',required=True)
    parser.add_argument('--delta',type=float,default=-.02)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    if not np.isfinite(args.delta) or not -.02<=args.delta<=-.005:parser.error('delta must be -0.005..-0.02 rad')
    import wujihandpy as sdk
    def interrupted(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupted)
    hand=sdk.Hand(serial_number=args.device)
    try:
        if int(hand.read_handedness())!=1:raise RuntimeError('Only first-generation LEFT hand supported')
        start=read_checked(hand)
        lower=vector(hand.read_joint_lower_limit());upper=vector(hand.read_joint_upper_limit())
        target,lo,hi,before,after=spacing_plan(start,lower,upper,args.delta)
        print(f'Index J2: {start[INDEX]:.6f} -> {target[INDEX]:.6f} rad; delta={args.delta} rad',flush=True)
        print(f'MODEL tip distance: {before*1000:.2f} -> {after*1000:.2f} mm (not measured clearance)',flush=True)
        if not args.apply:
            print('READ ONLY: no enable, no writes');return
        print('Only index J2 will be enabled; all other joints disabled. No return move. Starts in 5s; Ctrl+C cancels.',flush=True)
        for _ in range(5):
            current=read_checked(hand)
            if np.max(np.abs(current-start))>.01:raise RuntimeError('Starting pose changed during countdown')
            time.sleep(1)
        actual=apply_once(hand,sdk,start,target,lo,hi)
        print(f'Completed bounded adjustment; index J2 feedback={actual[INDEX]:.6f} rad. Hand disabled. Visually check spacing.',flush=True)
    finally:
        hand=None


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('Cancelled; no return motion')
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
