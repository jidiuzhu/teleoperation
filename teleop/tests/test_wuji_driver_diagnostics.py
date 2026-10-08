import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from utils.XHand_controller_wuji import (
    MAX_JOINT_EFFORT_LIMIT_A,
    WujiHandDriver,
    WujihandPyBackend,
    _configure_joint_effort_limits,
    _normalize_joint_effort_limits,
    _open_sdk_hand,
)
from utils.wuji_hand_spec import HAND_DOF, JOINT_LIMITS


class _FakeController:
    def __init__(self):
        self.positions = np.zeros((5, 4), dtype=np.float64)

    def set_joint_target_position(self, positions):
        self.positions = np.asarray(positions, dtype=np.float64).copy()

    def get_joint_actual_position(self):
        return self.positions.copy()


class _FakeHand:
    def read_input_voltage(self):
        return 24.0


class _FakeEffortLimitHand:
    def __init__(self, readback=None):
        self.readback = readback
        self.writes = []

    def write_joint_effort_limit(self, values):
        written = np.asarray(values, dtype=np.float64).copy()
        self.writes.append(written)
        if self.readback is None:
            self.readback = written

    def read_joint_effort_limit(self):
        return np.asarray(self.readback, dtype=np.float64).copy()


class _FakeInitHand(_FakeEffortLimitHand):
    def __init__(self):
        super().__init__()
        self.calls = []

    def read_handedness(self):
        return 0

    def read_joint_lower_limit(self):
        return JOINT_LIMITS[:, 0].reshape(5, 4)

    def read_joint_upper_limit(self):
        return JOINT_LIMITS[:, 1].reshape(5, 4)

    def read_joint_actual_position(self):
        return np.mean(JOINT_LIMITS, axis=1).reshape(5, 4)

    def get_product_sn(self):
        return "TEST-SN"

    def write_joint_effort_limit(self, values):
        self.calls.append("write_effort_limit")
        super().write_joint_effort_limit(values)

    def read_joint_effort_limit(self):
        self.calls.append("read_effort_limit")
        return super().read_joint_effort_limit()

    def disable_thread_safe_check(self):
        self.calls.append("disable_thread_safe_check")

    def write_joint_enabled(self, enabled):
        self.calls.append(f"enabled={enabled}")

    def realtime_controller(self, **_kwargs):
        self.calls.append("realtime_controller")
        return _FakeController()


class _FlakySdk:
    def __init__(self, failures):
        self.failures = failures
        self.calls = []
        self.hand = object()

    def Hand(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) <= self.failures:
            raise ConnectionError("transient descriptor ERROR_IO")
        return self.hand


def make_backend():
    backend = WujihandPyBackend.__new__(WujihandPyBackend)
    backend._lock = threading.Lock()
    backend._diagnostics_lock = threading.Lock()
    backend._diagnostics = {}
    backend._controller = _FakeController()
    backend._hand = _FakeHand()
    backend._closed = False
    backend._connection_error = None
    backend._heartbeat_interval_s = 0.0
    backend._last_heartbeat_time = 0.0
    backend._last_input_voltage = None
    backend._last_positions = np.zeros(HAND_DOF, dtype=np.float64)
    backend._joint_limits = JOINT_LIMITS.copy()
    return backend


class WujiDriverDiagnosticsTest(unittest.TestCase):
    def test_scalar_effort_limit_is_broadcast_to_every_joint(self):
        limits = _normalize_joint_effort_limits(1.0)

        self.assertEqual(limits.shape, (5, 4))
        np.testing.assert_array_equal(limits, np.ones((5, 4)))

    def test_effort_limit_rejects_unsafe_values_and_shapes(self):
        invalid = (0.0, -0.1, MAX_JOINT_EFFORT_LIMIT_A + 0.001, np.nan)
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _normalize_joint_effort_limits(value)

        with self.assertRaisesRegex(ValueError, "scalar or have SDK shape"):
            _normalize_joint_effort_limits(np.ones(20))

    def test_effort_limit_write_is_verified(self):
        hand = _FakeEffortLimitHand()

        actual = _configure_joint_effort_limits(hand, 1.25)

        np.testing.assert_array_equal(hand.writes[0], np.full((5, 4), 1.25))
        np.testing.assert_array_equal(actual, np.full((5, 4), 1.25))

    def test_effort_limit_readback_mismatch_fails(self):
        hand = _FakeEffortLimitHand(readback=np.full((5, 4), 1.0))

        with self.assertRaisesRegex(RuntimeError, "verification failed"):
            _configure_joint_effort_limits(hand, 1.25)

    def test_effort_limit_is_configured_before_motor_enable(self):
        hand = _FakeInitHand()
        sdk = SimpleNamespace(
            Hand=lambda **_kwargs: hand,
            filter=SimpleNamespace(LowPass=lambda **kwargs: kwargs),
        )

        with patch.dict("sys.modules", {"wujihandpy": sdk}):
            backend = WujihandPyBackend(device="TEST", effort_limit=1.0)

        try:
            self.assertLess(
                hand.calls.index("write_effort_limit"),
                hand.calls.index("enabled=True"),
            )
            self.assertEqual(backend.get_device_info()["joint_effort_limit_a"], 1.0)
        finally:
            backend.close()

    def test_mock_driver_reports_configured_effort_limit(self):
        driver = WujiHandDriver(backend_spec="mock", effort_limit=0.8)
        self.assertEqual(driver.get_device_info()["joint_effort_limit_a"], 0.8)

    def test_usb_open_retries_transient_constructor_failures(self):
        sdk = _FlakySdk(failures=2)
        sleeps = []

        hand = _open_sdk_hand(
            sdk,
            "347338703433",
            attempts=5,
            retry_delay_s=0.4,
            sleep=sleeps.append,
        )

        self.assertIs(hand, sdk.hand)
        self.assertEqual(len(sdk.calls), 3)
        self.assertEqual(
            sdk.calls, [{"serial_number": "347338703433"}] * 3
        )
        self.assertEqual(sleeps, [0.4, 0.4])

    def test_usb_open_stops_after_configured_attempts(self):
        sdk = _FlakySdk(failures=10)

        with self.assertRaisesRegex(ConnectionError, "after 3 attempts"):
            _open_sdk_hand(
                sdk,
                None,
                attempts=3,
                retry_delay_s=0.0,
                sleep=lambda _delay: None,
            )

        self.assertEqual(sdk.calls, [{"side": "right"}] * 3)

    def test_aggregate_reports_average_max_count_and_resets(self):
        backend = make_backend()
        backend._record_diagnostic("sample", 0.001)
        backend._record_diagnostic("sample", 0.003)

        diagnostics = backend.consume_diagnostics()

        self.assertEqual(diagnostics["sample"]["count"], 2)
        self.assertAlmostEqual(diagnostics["sample"]["avg_ms"], 2.0)
        self.assertAlmostEqual(diagnostics["sample"]["max_ms"], 3.0)
        self.assertEqual(backend.consume_diagnostics(), {})

    def test_send_and_read_emit_all_timing_stages(self):
        backend = make_backend()
        command = np.mean(JOINT_LIMITS, axis=1)

        backend.send_joint_positions(command)
        feedback = backend.read_joint_positions()
        diagnostics = backend.consume_diagnostics()

        np.testing.assert_allclose(feedback, command)
        expected = {
            "send_lock_wait",
            "send_pdo",
            "send_total",
            "read_lock_wait",
            "read_pdo",
            "read_total",
            "heartbeat_sdo",
        }
        self.assertEqual(set(diagnostics), expected)
        for stat in diagnostics.values():
            self.assertEqual(stat["count"], 1)
            self.assertGreaterEqual(stat["avg_ms"], 0.0)
            self.assertGreaterEqual(stat["max_ms"], stat["avg_ms"])


if __name__ == "__main__":
    unittest.main()
