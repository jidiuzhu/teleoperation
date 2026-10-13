"""Reproduce the vendor asynchronous send buffer without hardware."""
import unittest
from unittest.mock import Mock, patch
from utils.marvin_controller import _NativeSDK


class NativeTransactionsTest(unittest.TestCase):
    def native(self):
        sdk = _NativeSDK.__new__(_NativeSDK)
        sdk._limits = Mock(return_value=True)
        sdk._target = Mock(return_value=True)
        sdk._mode = Mock(return_value=True)
        sdk._send = Mock(return_value=True)
        return sdk

    def test_mode_waits_for_pending_initial_target_without_duplicate_send(self):
        sdk = self.native()
        sdk._clear = Mock(side_effect=[True, False, False, True])
        with patch('utils.marvin_controller.time.sleep'):
            sdk.configure(1,1,[0]*7)
        self.assertEqual(sdk._clear.call_count,4)
        sdk._limits.assert_called_once_with(1,1)
        sdk._target.assert_called_once()
        sdk._mode.assert_called_once_with(1)
        self.assertEqual(sdk._send.call_count,2)

    def test_busy_timeout_never_stages_mode(self):
        sdk = self.native()
        sdk._clear = Mock(side_effect=[True,False])
        with patch('utils.marvin_controller.time.monotonic',side_effect=[0,0,0.101]):
            with self.assertRaises(TimeoutError): sdk.configure(1,1,[0]*7)
        sdk._mode.assert_not_called()
        self.assertEqual(sdk._send.call_count,1)

    def test_stream_command_waits_and_submits_only_once(self):
        sdk = self.native();sdk._clear=Mock(side_effect=[False,True])
        with patch('utils.marvin_controller.time.sleep'):
            self.assertTrue(sdk.command([0]*7))
        sdk._target.assert_called_once();sdk._send.assert_called_once()

    def test_send_failure_does_not_retry_or_enable(self):
        sdk = self.native();sdk._clear=Mock(return_value=True);sdk._send.return_value=False
        with self.assertRaises(RuntimeError):sdk.configure(1,1,[0]*7)
        sdk._mode.assert_not_called();sdk._send.assert_called_once()


if __name__=='__main__':unittest.main()
