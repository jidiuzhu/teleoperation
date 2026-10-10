"""Official wujihandpy driver for the Wuji real-data teleoperation path."""

from importlib import import_module
from pathlib import Path
import threading
import time

import numpy as np

from utils.wuji_hand_spec import (
    HAND_DOF,
    JOINT_LIMITS,
    JOINT_NAMES,
    validate_joint_vector,
    hand_spec,
)


_SDK_ARRAY_SHAPE = (5, 4)
_RIGHT_HANDEDNESS = 0
_CONNECT_ATTEMPTS = 5
_CONNECT_RETRY_DELAY_S = 0.4
MAX_JOINT_EFFORT_LIMIT_A = 3.5
_EFFORT_LIMIT_VERIFY_ATOL_A = 1e-3


def _flatten_sdk_array(values, *, label):
    array = np.asarray(values, dtype=np.float64)
    if array.shape != _SDK_ARRAY_SHAPE:
        raise ValueError(
            f"{label} must have SDK shape {_SDK_ARRAY_SHAPE}, got {array.shape}"
        )
    return validate_joint_vector(array.reshape(HAND_DOF), label=label)


def _normalize_joint_effort_limits(values):
    """Return a validated (5, 4) SDK effort-limit matrix in amperes."""
    limits = np.asarray(values, dtype=np.float64)
    if limits.ndim == 0:
        limits = np.full(_SDK_ARRAY_SHAPE, float(limits), dtype=np.float64)
    elif limits.shape != _SDK_ARRAY_SHAPE:
        raise ValueError(
            "Wuji joint effort limits must be a scalar or have SDK shape "
            f"{_SDK_ARRAY_SHAPE}, got {limits.shape}"
        )
    if not np.all(np.isfinite(limits)):
        raise ValueError("Wuji joint effort limits must be finite")
    if np.any(limits <= 0.0) or np.any(limits > MAX_JOINT_EFFORT_LIMIT_A):
        raise ValueError(
            "Wuji joint effort limits must be within "
            f"(0, {MAX_JOINT_EFFORT_LIMIT_A}] A"
        )
    return limits.copy()


def _configure_joint_effort_limits(hand, values):
    """Write and verify all Wuji joint effort limits before motor enable."""
    requested = _normalize_joint_effort_limits(values)
    hand.write_joint_effort_limit(requested)
    actual = _normalize_joint_effort_limits(hand.read_joint_effort_limit())
    if not np.allclose(
        actual,
        requested,
        rtol=0.0,
        atol=_EFFORT_LIMIT_VERIFY_ATOL_A,
    ):
        max_error = float(np.max(np.abs(actual - requested)))
        raise RuntimeError(
            "Wuji joint effort-limit verification failed: "
            f"maximum readback error is {max_error:.6f} A"
        )
    return actual


def _open_sdk_hand(
    sdk,
    device,
    *,
    side="right",
    attempts=_CONNECT_ATTEMPTS,
    retry_delay_s=_CONNECT_RETRY_DELAY_S,
    sleep=time.sleep,
):
    """Open the hand, retrying only before motors or realtime control exist."""
    if attempts < 1:
        raise ValueError("Wuji connection attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            if device:
                return sdk.Hand(serial_number=str(device))
            return sdk.Hand(side=side)
        except ConnectionError as exc:
            if attempt == attempts:
                raise ConnectionError(
                    f"Wuji USB connection failed after {attempts} attempts"
                ) from exc
            print(
                f"[WujiUSB] Connection attempt {attempt}/{attempts} failed: "
                f"{exc}; retrying in {retry_delay_s:.1f}s...",
                flush=True,
            )
            sleep(retry_delay_s)


class WujihandPyBackend:
    """Control a first-generation Wuji Hand through its official USB SDK."""

    def __init__(
        self,
        device=None,
        joint_names=JOINT_NAMES,
        *,
        effort_limit=None,
        lowpass_cutoff_hz=5.0,
        heartbeat_interval_s=1.0,
        side="right",
        enable_motion=True,
        defer_enable=False,
    ):
        if tuple(joint_names) != JOINT_NAMES:
            raise ValueError(
                "wujihandpy uses firmware order thumb-to-little, J1-to-J4; "
                f"unexpected joint order: {tuple(joint_names)}"
            )
        if lowpass_cutoff_hz <= 0:
            raise ValueError("lowpass_cutoff_hz must be positive")
        if heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be positive")
        requested_effort_limits = (
            None
            if effort_limit is None
            else _normalize_joint_effort_limits(effort_limit)
        )

        try:
            import wujihandpy
        except ImportError as exc:
            source_checkout = Path(__file__).resolve().parents[2] / "wujihandpy-main"
            source_note = ""
            if source_checkout.is_dir():
                source_note = (
                    f" The source checkout at {source_checkout} is present, but "
                    "wujihandpy contains a compiled _core extension and cannot be "
                    "imported directly from its source directory."
                )
            raise RuntimeError(
                "wujihandpy is not installed in this Python environment. Install "
                "the official wheel with 'python -m pip install "
                "wujihandpy==1.8.0'." + source_note
            ) from exc

        hand_spec(side)
        self._lowpass_cutoff_hz = float(lowpass_cutoff_hz)
        self._motion_allowed = bool(enable_motion)
        self._side = side
        self._sdk = wujihandpy
        self._lock = threading.Lock()
        self._diagnostics_lock = threading.Lock()
        self._diagnostics = {}
        self._hand = None
        self._controller = None
        self._enabled = False
        self._closed = False
        self._connection_error = None
        self._heartbeat_interval_s = float(heartbeat_interval_s)
        self._last_heartbeat_time = 0.0
        self._last_input_voltage = None
        self._last_positions = np.zeros(HAND_DOF, dtype=np.float64)
        self._joint_limits = JOINT_LIMITS.copy()
        self._device_info = {}

        try:
            # Select and verify the requested side independently of the USB serial.
            self._hand = _open_sdk_hand(wujihandpy, device, side=side)

            handedness = int(self._hand.read_handedness())
            if handedness != (0 if side == "right" else 1):
                raise RuntimeError(
                    f"Connected Wuji device reports handedness={handedness}; "
                    f"the selected teleoperation path requires a {side} hand."
                )

            lower = _flatten_sdk_array(
                self._hand.read_joint_lower_limit(), label="Wuji hardware lower limits"
            )
            upper = _flatten_sdk_array(
                self._hand.read_joint_upper_limit(), label="Wuji hardware upper limits"
            )
            if np.any(lower >= upper):
                raise RuntimeError("Wuji hardware returned invalid joint limits")
            self._joint_limits = np.stack([lower, upper], axis=1)
            self._last_positions = _flatten_sdk_array(
                self._hand.read_joint_actual_position(), label="Wuji initial feedback"
            )

            try:
                product_sn = self._hand.get_product_sn()
            except Exception:
                product_sn = "unknown"
            self._device_info = {
                "usb_serial_number": str(device) if device else "auto-by-side",
                "product_serial_number": product_sn,
                "handedness": side,
                "transport": "USB-CDC",
                "sdk": "wujihandpy",
            }

            if requested_effort_limits is not None and self._motion_allowed:
                verified_effort_limits = _configure_joint_effort_limits(
                    self._hand, requested_effort_limits
                )
                self._device_info["joint_effort_limit_a"] = float(
                    verified_effort_limits[0, 0]
                )
                print(
                    "[WujiSafety] Joint effort limit configured and verified: "
                    f"{verified_effort_limits[0, 0]:.3f} A for all {HAND_DOF} joints",
                    flush=True,
                )

            # The server reads feedback in a background thread and writes commands
            # in the control thread. The SDK requires an external mutex in this mode.
            self._hand.disable_thread_safe_check()
            # Mark it before the blocking write so cleanup still attempts to
            # disable if the USB acknowledgement times out after motors changed state.
            if self._motion_allowed and not defer_enable:
                self.enable()
            self._last_heartbeat_time = time.monotonic()
        except Exception:
            self.close()
            raise

    def enable(self):
        with self._lock:
            self._raise_if_unavailable()
            if not self._motion_allowed:
                raise RuntimeError("Hand motion is disabled")
            if self._enabled:
                return
            current = _flatten_sdk_array(self._hand.read_joint_actual_position(), label="Wuji enable feedback")
            try:
                self._controller = self._hand.realtime_controller(
                    enable_upstream=True,
                    filter=self._sdk.filter.LowPass(cutoff_freq=self._lowpass_cutoff_hz),
                )
                self._controller.set_joint_target_position(current.reshape(_SDK_ARRAY_SHAPE))
                self._enabled = True
                self._hand.write_joint_enabled(True)
            except Exception:
                # Caller closes backend; mark enabled before acknowledgement for cleanup.
                self._enabled = True
                raise

    def _raise_if_unavailable(self):
        if self._closed or self._hand is None:
            raise ConnectionError("Wuji Hand controller is closed")
        if self._connection_error is not None:
            raise ConnectionError("Wuji Hand USB connection was lost") from self._connection_error

    def _record_diagnostic(self, name, elapsed_s):
        elapsed_ms = float(elapsed_s) * 1000.0
        with self._diagnostics_lock:
            stat = self._diagnostics.setdefault(
                name, {"total_ms": 0.0, "max_ms": 0.0, "count": 0}
            )
            stat["total_ms"] += elapsed_ms
            stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
            stat["count"] += 1

    def consume_diagnostics(self):
        """Return and reset aggregate timings without taking the SDK I/O lock."""
        with self._diagnostics_lock:
            raw = self._diagnostics
            self._diagnostics = {}
        return {
            name: {
                "avg_ms": stat["total_ms"] / stat["count"],
                "max_ms": stat["max_ms"],
                "count": stat["count"],
            }
            for name, stat in raw.items()
            if stat["count"]
        }

    def _heartbeat_if_due(self):
        now = time.monotonic()
        if now - self._last_heartbeat_time < self._heartbeat_interval_s:
            return
        started = time.perf_counter()
        try:
            try:
                # Realtime SDK calls use a cache and do not report USB removal. A
                # synchronous low-rate read is the vendor-recommended disconnect probe.
                self._last_input_voltage = float(self._hand.read_input_voltage())
                self._last_heartbeat_time = now
            except OSError as exc:
                self._connection_error = exc
                raise ConnectionError("Wuji Hand USB heartbeat failed") from exc
        finally:
            self._record_diagnostic(
                "heartbeat_sdo", time.perf_counter() - started
            )

    def send_joint_positions(self, joint_positions):
        if not self._motion_allowed:
            raise RuntimeError("Hand motion is disabled")
        joints = validate_joint_vector(joint_positions, label="Wuji SDK command")
        lower = self._joint_limits[:, 0]
        upper = self._joint_limits[:, 1]
        if np.any(joints < lower) or np.any(joints > upper):
            bad = np.flatnonzero((joints < lower) | (joints > upper)).tolist()
            raise ValueError(f"Wuji command exceeds calibrated hardware limits at {bad}")

        started = time.perf_counter()
        lock_started = started
        lock_wait_s = 0.0
        try:
            with self._lock:
                lock_wait_s = time.perf_counter() - lock_started
                self._raise_if_unavailable()
                pdo_started = time.perf_counter()
                try:
                    self._controller.set_joint_target_position(
                        joints.reshape(_SDK_ARRAY_SHAPE)
                    )
                finally:
                    self._record_diagnostic(
                        "send_pdo", time.perf_counter() - pdo_started
                    )
        finally:
            self._record_diagnostic("send_lock_wait", lock_wait_s)
            self._record_diagnostic("send_total", time.perf_counter() - started)

    def read_joint_positions(self):
        started = time.perf_counter()
        lock_started = started
        lock_wait_s = 0.0
        try:
            with self._lock:
                lock_wait_s = time.perf_counter() - lock_started
                self._raise_if_unavailable()
                self._heartbeat_if_due()
                pdo_started = time.perf_counter()
                try:
                    self._last_positions = _flatten_sdk_array(
                        (self._controller.get_joint_actual_position() if self._controller is not None
                         else self._hand.read_joint_actual_position()),
                        label="Wuji realtime feedback",
                    )
                finally:
                    self._record_diagnostic(
                        "read_pdo", time.perf_counter() - pdo_started
                    )
                return self._last_positions.copy()
        finally:
            self._record_diagnostic("read_lock_wait", lock_wait_s)
            self._record_diagnostic("read_total", time.perf_counter() - started)

    def get_joint_limits(self):
        return self._joint_limits.copy()

    def get_device_info(self):
        info = dict(self._device_info)
        info["input_voltage"] = self._last_input_voltage
        return info

    def close(self):
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            if self._closed:
                return
            self._closed = True

            controller = self._controller
            self._controller = None
            if controller is not None:
                try:
                    controller.close()
                except Exception:
                    pass

            hand = self._hand
            self._hand = None
            if hand is not None and self._enabled:
                try:
                    hand.write_joint_enabled(False)
                except Exception:
                    pass
            self._enabled = False


class MockWujiBackend:
    """In-memory backend for offline integration tests; selected explicitly."""

    def __init__(self, device=None, joint_names=JOINT_NAMES, effort_limit=None, side="right", enable_motion=True, defer_enable=False):
        self._limits = hand_spec(side)[2]
        self._motion_allowed = enable_motion
        self.joint_names = tuple(joint_names)
        self.q = np.zeros(len(self.joint_names), dtype=np.float64)
        self.effort_limits = (
            None
            if effort_limit is None
            else _normalize_joint_effort_limits(effort_limit)
        )

    def send_joint_positions(self, joint_positions):
        if not self._motion_allowed:
            raise RuntimeError("Hand motion is disabled")
        self.q = validate_joint_vector(joint_positions, label="Mock Wuji command").copy()

    def read_joint_positions(self):
        return self.q.copy()

    def get_joint_limits(self):
        return self._limits.copy()

    def get_device_info(self):
        info = {"sdk": "mock", "transport": "memory"}
        if self.effort_limits is not None:
            info["joint_effort_limit_a"] = float(self.effort_limits[0, 0])
        return info

    def consume_diagnostics(self):
        return {}

    def close(self):
        pass


def _load_backend(spec: str):
    try:
        module_name, class_name = spec.split(":", 1)
    except ValueError as exc:
        raise ValueError("Wuji backend must use the form 'python.module:BackendClass'") from exc
    module = import_module(module_name)
    return getattr(module, class_name)


class WujiHandDriver:
    """Expose the hand interface expected by the Wuji teleoperation server."""

    def __init__(self, device=None, backend_spec=None, effort_limit=None, *, side="right", enable_motion=True, defer_enable=False):
        if backend_spec in (None, "wujihandpy"):
            backend_cls = WujihandPyBackend
        elif backend_spec == "mock":
            backend_cls = MockWujiBackend
        else:
            backend_cls = _load_backend(backend_spec)
        self._motion_allowed = enable_motion
        backend_kwargs = {"device": device, "joint_names": JOINT_NAMES}
        if backend_spec in (None, "wujihandpy", "mock") or side != "right" or not enable_motion:
            backend_kwargs.update(side=side, enable_motion=enable_motion, defer_enable=defer_enable)
        if effort_limit is not None:
            backend_kwargs["effort_limit"] = effort_limit
        self._backend = backend_cls(**backend_kwargs)

    def enable(self):
        if not self._motion_allowed:
            raise RuntimeError("Hand motion is disabled")
        if hasattr(self._backend, "enable"):
            self._backend.enable()

    def send_joints(self, joint_vals):
        if not self._motion_allowed:
            raise RuntimeError("Hand motion is disabled")
        joints = validate_joint_vector(joint_vals, label="Wuji command")
        self._backend.send_joint_positions(joints.copy())

    def get_current_joints(self):
        joints = self._backend.read_joint_positions()
        return validate_joint_vector(joints, label="Wuji feedback")

    def get_joint_limits(self):
        if not hasattr(self._backend, "get_joint_limits"):
            return JOINT_LIMITS.copy()
        limits = np.asarray(self._backend.get_joint_limits(), dtype=np.float64)
        if limits.shape != (HAND_DOF, 2) or not np.all(np.isfinite(limits)):
            raise ValueError(f"Wuji joint limits must have shape ({HAND_DOF}, 2)")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("Wuji joint limits contain lower >= upper")
        return limits.copy()

    def get_device_info(self):
        if hasattr(self._backend, "get_device_info"):
            return dict(self._backend.get_device_info())
        return {}

    def consume_diagnostics(self):
        if hasattr(self._backend, "consume_diagnostics"):
            return dict(self._backend.consume_diagnostics())
        return {}

    def get_composite_state(self):
        joints = self.get_current_joints()
        tactile = self._backend.read_tactile() if hasattr(self._backend, "read_tactile") else {}
        return {"joints": joints, "tactile": tactile or {}}

    def reset_tactile_sensors(self):
        if hasattr(self._backend, "reset_tactile_sensors"):
            self._backend.reset_tactile_sensors()

    def close(self):
        self._backend.close()


__all__ = [
    "HAND_DOF",
    "JOINT_NAMES",
    "MAX_JOINT_EFFORT_LIMIT_A",
    "MockWujiBackend",
    "WujihandPyBackend",
    "WujiHandDriver",
]
