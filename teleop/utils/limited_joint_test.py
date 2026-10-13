"""Finite single-joint commissioning trajectory. No hardware imports here."""
import math
import json
import threading
import time
import numpy as np


def plan_single_joint(start, joint, delta, validate):
    start = np.asarray(start, dtype=float)
    if start.ndim != 1 or not np.isfinite(start).all():
        raise ValueError('Invalid starting feedback')
    if not 0 <= joint < len(start) or not math.isfinite(delta):
        raise ValueError('Invalid joint or displacement')
    target = start.copy(); target[joint] += delta
    # J6/J7 coupled bounds must also hold between endpoints.
    for fraction in np.linspace(0, 1, 101):
        validate(start + fraction*(target-start))
    return start.copy(), target


def run_single_joint(device, start, target, validate, *, duration=10.0,
                     tracking_tolerance, completion_tolerance, feedback_speed_limit,
                     clock=time.monotonic, sleep=time.sleep):
    """One cosine excursion and return; exceptions stop, never attempt recovery motion.

    device: read(), enable(), send(q), stop(). read() must query live hardware.
    The watchdog requests stop if the control loop misses its 0.3s deadline.
    It cannot replace a hardware emergency stop for a blocked SDK/USB call.
    """
    if not math.isfinite(duration) or not 8 <= duration <= 20:
        raise ValueError('Test duration must be 8..20 seconds')
    start, target = np.asarray(start,float), np.asarray(target,float)
    if start.shape != target.shape or start.ndim != 1 or not np.isfinite(np.r_[start,target]).all():
        raise ValueError('Invalid trajectory endpoints')
    moving = np.flatnonzero(np.abs(target-start) > 1e-10)
    if len(moving) != 1:
        raise ValueError('Test must change exactly one joint')
    joint = int(moving[0])
    amplitude = abs(target[joint]-start[joint])
    direction = np.sign(target[joint]-start[joint])
    observed_excursion = 0.0
    done = threading.Event()
    pulse = [time.monotonic()]
    faults = []

    def watchdog():
        while not done.wait(.025):
            if time.monotonic()-pulse[0] > .3:
                faults.append('control loop watchdog expired')
                try: device.stop()
                except Exception as exc: faults.append(str(exc))
                return

    monitor = None
    try:
        actual = np.asarray(device.read(),float)
        validate(actual)
        if actual.shape != start.shape or np.max(np.abs(actual-start)) > completion_tolerance:
            raise RuntimeError('Starting pose changed; rerun the read-only plan')
        enable_started = clock()
        device.enable()
        enable_duration = clock()-enable_started
        pulse[0] = time.monotonic()
        monitor = threading.Thread(target=watchdog,daemon=True); monitor.start()
        began = previous_time = clock()
        previous_feedback = actual.copy()
        previous_target = start.copy()
        settled = 0
        while True:
            now = clock(); elapsed = now-began
            if faults: raise RuntimeError('; '.join(faults))
            if now-previous_time > .2:
                raise TimeoutError('Control loop stalled; stopping without return motion')
            read_started = clock()
            actual = np.asarray(device.read(),float)
            now = clock()
            if now-read_started > .2:
                raise TimeoutError('Feedback query exceeded 0.2 seconds')
            if actual.shape != start.shape:
                raise ValueError('Feedback shape changed')
            validate(actual)
            dt = max(now-previous_time,1e-6)
            if np.max(np.abs(actual-previous_target)) > tracking_tolerance:
                errors = actual-previous_target
                bad = np.flatnonzero(np.abs(errors)>tracking_tolerance)
                details = []
                for index in bad:
                    label = f"joint {index+1}"
                    if len(actual) == 20:
                        label += f" (finger{index//4+1}_joint{index%4+1})"
                    details.append(f"{label}: actual={actual[index]:+.6f}, target={previous_target[index]:+.6f}, error={errors[index]:+.6f}")
                snapshot = {"elapsed_s": float(elapsed), "enable_duration_s": float(enable_duration),
                            "pre_enable_actual": previous_feedback.tolist() if elapsed < 0.01 else None,
                            "selected_joint": joint+1,
                            "threshold": tracking_tolerance, "actual": actual.tolist(),
                            "previous_target": previous_target.tolist(), "start": start.tolist()}
                print("[TRACKING FAILURE] " + json.dumps(snapshot), flush=True)
                raise RuntimeError(
                    f"Hardware tracking error exceeded test threshold {tracking_tolerance} "
                    f"at {elapsed:.3f}s; " + "; ".join(details))
            if np.max(np.abs(actual-previous_feedback)) > feedback_speed_limit*dt + completion_tolerance:
                raise RuntimeError('Unexpected feedback jump')
            observed_excursion = max(observed_excursion, direction*(actual[joint]-start[joint]))
            # Bound every axis around the whole planned excursion, not only the selected axis.
            if np.any(actual < np.minimum(start,target)-tracking_tolerance) or np.any(actual > np.maximum(start,target)+tracking_tolerance):
                raise RuntimeError('Hardware left test envelope')
            fraction = .5*(1-math.cos(2*math.pi*min(elapsed,duration)/duration))
            command = start + fraction*(target-start)
            validate(command)
            if faults: raise RuntimeError('; '.join(faults))
            device.send(command)
            pulse[0] = time.monotonic()
            if elapsed >= duration:
                settled = settled+1 if np.max(np.abs(actual-start)) <= completion_tolerance else 0
                if settled >= 5:
                    if observed_excursion < amplitude*0.5:
                        raise RuntimeError('No sufficient measured joint excursion; commands alone do not prove motion')
                    return
                if elapsed >= duration+2:
                    raise TimeoutError('Did not settle at starting pose')
            previous_feedback, previous_target, previous_time = actual.copy(), command, now
            sleep(.05)
    finally:
        done.set()
        try:
            device.stop()
        finally:
            if monitor is not None: monitor.join(timeout=.5)
