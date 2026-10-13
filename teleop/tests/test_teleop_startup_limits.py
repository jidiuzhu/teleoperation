import threading
import unittest
from unittest.mock import Mock, patch
import numpy as np
from utils.marvin_controller import MarvinController
from tests.test_marvin_adaptation import FakeArm
from scripts.teleop_real_server_wuji import AsyncHardwareInterface

class StartupLimitTests(unittest.TestCase):
    def test_target_shaping_preserves_existing_speed_bound(self):
        sdk=FakeArm();arm=MarvinController('192.0.2.1',enable_motion=True,_backend=sdk)
        try:
            arm.enable();start=arm.last_command.copy()
            requested=start.copy();requested[:2]+=[4,2]
            stamp=arm.last_command_time
            with patch('utils.marvin_controller.time.monotonic',return_value=stamp+.02):
                limited=arm.limit_joint_target(requested)
            np.testing.assert_allclose(limited[:2]-start[:2],[.4,.2],atol=1e-7)
            self.assertEqual(len([e for e in sdk.events if isinstance(e,tuple) and e[0]=='command']),0)
            # strict command API still rejects unsmoothed targets
            with self.assertRaises(ValueError):arm.move_j_stream(requested)
        finally:arm.disconnect()

    def test_shaping_retains_coupled_limit_rejection(self):
        sdk=FakeArm();arm=MarvinController('192.0.2.1',enable_motion=True,_backend=sdk)
        try:
            arm.enable();q=arm.last_command.copy();q[5:]=[60,80]
            with self.assertRaises(ValueError):arm.limit_joint_target(q)
        finally:arm.disconnect()

    def test_feedback_timestamp_precedes_slow_camera_and_failed_read_is_invalid(self):
        hw=AsyncHardwareInterface.__new__(AsyncHardwareInterface)
        hw.lock=threading.Lock();hw.feedback_io_lock=threading.Lock()
        hw.diagnostics_lock=threading.Lock();hw.diagnostics={}
        hw.cache={'arm_state':None,'hand_state':None,'frames':None}
        hw.last_arm_success_time=hw.last_hand_success_time=None
        hw.arm=Mock();hw.arm.get_joint_state.return_value=(True,[1]*7)
        hw.hand=Mock();hw.hand.get_composite_state.return_value={'joints':np.zeros(20)}
        hw.camera=Mock();hw.camera.get_frames.return_value={}
        with patch('scripts.teleop_real_server_wuji.time.monotonic',side_effect=[1.0,1.1]):
            hw._update_once()
        snap=hw.get_snapshot();self.assertEqual(snap[3:],(1.0,1.1));self.assertEqual(snap[0],[1]*7)
        hw.arm.get_joint_state.return_value=(False,[])
        with patch('scripts.teleop_real_server_wuji.time.monotonic',side_effect=[2.0,2.1]):
            hw.refresh_joint_feedback()
        snap=hw.get_snapshot();self.assertIsNone(snap[0]);self.assertEqual(snap[3],1.0)

if __name__=='__main__':unittest.main()
