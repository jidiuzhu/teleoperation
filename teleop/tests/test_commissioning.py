import contextlib
import io
import time
import unittest
from unittest.mock import Mock
import numpy as np
from utils.limited_joint_test import plan_single_joint, run_single_joint
from scripts.test_single_device import parse_args
from scripts.teleop_real_server_wuji import TeleopServer
from utils.marvin_kinematics import MarvinKinematics, VENDOR_ROOT


class Clock:
    def __init__(self):self.t=0
    def __call__(self):return self.t
    def sleep(self,dt):self.t+=dt


class Device:
    def __init__(self,q):
        self.q=np.array(q,float);self.commands=[];self.stops=0;self.enabled=False;self.fail=False
    def read(self):
        if self.fail:raise ConnectionError('lost feedback')
        return self.q.copy()
    def enable(self):self.enabled=True
    def send(self,q):self.commands.append(q.copy());self.q=q.copy()
    def stop(self):self.stops+=1;self.enabled=False


def validate(q):
    if not np.isfinite(q).all() or np.max(np.abs(q))>3:raise ValueError('bounds')


class SingleJointTests(unittest.TestCase):
    def run_device(self,device,clock):
        start,target=plan_single_joint([0,0,0],1,1,validate)
        run_single_joint(device,start,target,validate,duration=8,
                         tracking_tolerance=.1,completion_tolerance=.01,
                         feedback_speed_limit=2,clock=clock,sleep=clock.sleep)
    def test_one_axis_excursion_returns_and_stops(self):
        d=Device([0,0,0]);self.run_device(d,Clock())
        commands=np.array(d.commands)
        self.assertGreater(commands[:,1].max(),.99)
        np.testing.assert_array_equal(commands[:,[0,2]],0)
        np.testing.assert_allclose(d.q,0,atol=.01)
        self.assertFalse(d.enabled);self.assertEqual(d.stops,1)
    def test_feedback_failure_stops_without_recovery_commands(self):
        d=Device([0,0,0]);clock=Clock()
        original=d.send
        def send(q):original(q);d.fail=True
        d.send=send
        with self.assertRaises(ConnectionError):self.run_device(d,clock)
        self.assertEqual(len(d.commands),1);self.assertEqual(d.stops,1)
    def test_stall_stops_without_catching_up(self):
        d=Device([0,0,0]);clock=Clock();clock.sleep=lambda dt:setattr(clock,'t',clock.t+.5)
        with self.assertRaises(TimeoutError):self.run_device(d,clock)
        self.assertEqual(len(d.commands),1);self.assertEqual(d.stops,1)
    def test_unexpected_feedback_jump_stops(self):
        d=Device([0,0,0]);clock=Clock()
        original=d.send
        def send(q):original(q);d.q[0]=1
        d.send=send
        with self.assertRaises(RuntimeError):self.run_device(d,clock)
        self.assertEqual(len(d.commands),1);self.assertEqual(d.stops,1)
    def test_tracking_failure_identifies_unselected_joint_and_stops(self):
        d=Device([0,0,0]);clock=Clock()
        def send(q):
            d.commands.append(q.copy());d.q=q.copy();d.q[2]=.5
        d.send=send
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, r'joint 3: actual=\+0.500000'):
                self.run_device(d,clock)
        self.assertIn('"selected_joint": 2',output.getvalue())
        self.assertEqual(d.stops,1)
        self.assertEqual(len(d.commands),1)

    def test_interruption_stops(self):
        d=Device([0,0,0]);clock=Clock()
        def interrupt(dt):raise KeyboardInterrupt()
        clock.sleep=interrupt
        with self.assertRaises(KeyboardInterrupt):self.run_device(d,clock)
        self.assertEqual(d.stops,1)
    def test_stationary_feedback_is_not_reported_as_success(self):
        d=Device([0,0,0]);d.send=lambda q:d.commands.append(q.copy())
        start,target=plan_single_joint([0,0,0],1,.02,validate)
        clock=Clock()
        with self.assertRaisesRegex(RuntimeError,'excursion'):
            run_single_joint(d,start,target,validate,duration=8,tracking_tolerance=.1,
                completion_tolerance=.01,feedback_speed_limit=2,clock=clock,sleep=clock.sleep)
        self.assertEqual(d.stops,1)

    def test_preflight_checks_path_before_enabling(self):
        with self.assertRaises(ValueError):plan_single_joint([0,0,0],1,4,validate)
    def test_cli_rejects_missing_confirmation_and_excessive_delta(self):
        base=['arm','--arm-ip','192.0.2.1','--joint','1']
        self.assertFalse(parse_args(base).enable_motion)
        with contextlib.redirect_stderr(io.StringIO()):
            for extra in [['--enable-motion'],['--delta','3'],['--duration','nan'],['--confirmation','MOVE_HAND']]:
                with self.assertRaises(SystemExit):parse_args(base+extra)
        self.assertTrue(parse_args(base+['--enable-motion','--confirmation','MOVE_ARM']).enable_motion)


@unittest.skipUnless((VENDOR_ROOT/'SDK_PYTHON/libKine.so').exists(),'Build SDK first')
class PreviewTests(unittest.TestCase):
    def server(self):
        s=TeleopServer.__new__(TeleopServer)
        s.preview_targets=True;s.motion_enabled=False;s.teleop_enabled=False
        s.arm=Mock();s.hand=Mock()
        s.arm.get_joint_state.return_value=(True,[21.8,-41,-4.74,-63.67,10.15,14.72,7.68])
        s.hand.get_current_joints.return_value=np.zeros(20)
        s.kinematics=MarvinKinematics();s.flange_thickness=0
        s.max_linear_speed=.05;s.max_angular_speed=20
        s.hand_limits_min=np.full(20,-1);s.hand_limits_max=np.ones(20)
        s._preview_print_time=0;s._finish_recording=Mock()
        s._enable_teleop()
        return s
    def test_preview_computes_ik_without_enabling_or_writing(self):
        s=self.server();s.desired_ee_pose[0]+=.005;s.hand_target_q[:]=.1
        for _ in range(8):s._step_control(time.time(),{'arm_qpos':s.arm_curr_q_real_rad})
        self.assertTrue(s.last_preview['ik']['success'])
        self.assertGreater(s.last_preview['requested_delta_m'][0],0)
        self.assertFalse(s.last_preview['motion_sent'])
        s.arm.enable.assert_not_called();s.hand.enable.assert_not_called()
        s.arm.move_j_stream.assert_not_called();s.hand.send_joints.assert_not_called()
        s._stop_teleop('test')
        s.arm.stop.assert_not_called()
    def test_failed_preview_ik_stops_without_targets(self):
        s=self.server();s.kinematics.solve_ik=Mock(return_value=s.arm_target_q_rad)
        s.kinematics.last_solve_info={'success':False,'status':'unreachable'}
        s._step_control(time.time(),{'arm_qpos':s.arm_curr_q_real_rad})
        self.assertFalse(s.teleop_enabled)
        self.assertIsNone(s.last_preview['arm_candidate_deg'])
        s.arm.move_j_stream.assert_not_called();s.hand.send_joints.assert_not_called()
    def test_preview_stale_input_requires_reanchor(self):
        s=self.server();s.last_command_time=time.monotonic()-1
        s._expire_teleop_if_stale()
        self.assertFalse(s.teleop_enabled);self.assertIsNone(s.human_anchor_pos)
        s.arm.enable.assert_not_called();s.arm.stop.assert_not_called()
    def test_conflicting_modes_fail_before_hardware(self):
        with self.assertRaisesRegex(ValueError,'mutually exclusive'):
            TeleopServer('x','y','mock',None,0,'none',enable_motion=True,preview_targets=True)


if __name__=='__main__':unittest.main()
