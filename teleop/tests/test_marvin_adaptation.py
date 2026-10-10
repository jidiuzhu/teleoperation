"""Behavioral checks for MARVIN SDK boundaries and fail-closed teleoperation."""
import ctypes as C
import sys
import time
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from utils.marvin_controller import MarvinController, _NativeSDK
from utils.marvin_kinematics import MarvinModel, MarvinKinematics, VENDOR_ROOT
from utils.wuji_hand_spec import hand_spec
from utils.XHand_controller_wuji import WujihandPyBackend
from scripts.teleop_real_server_wuji import TeleopServer
from tests.test_wuji_driver_diagnostics import _FakeInitHand


class FakeArm:
    def __init__(self):
        self.frame=0; self.frozen=False; self.mode=0
        self.q=[21.8,-41,-4.74,-63.67,10.15,14.72,7.68]
        self.events=[]
    def connect(self, ip): self.events.append('connect'); return True
    def version(self): return 100341
    def read(self):
        if not self.frozen: self.frame+=1
        out=dict(frame_serial=self.frame,fb_joint_pos=self.q)
        state=dict(err_code=0,cur_state=self.mode)
        return dict(outputs=[out,out],states=[state,state])
    def configure(self,vel,acc,q):
        self.events.append(('enable',q)); self.mode=1
    def command(self,q): self.events.append(('command',q)); self.q=q; return True
    def stop(self): self.events.append('stop'); self.mode=0
    def release(self): self.events.append('release')


class ControllerTests(unittest.TestCase):
    def make_arm(self, **kwargs):
        sdk=FakeArm()
        arm=MarvinController('192.0.2.1',_backend=sdk,**kwargs)
        self.addCleanup(arm.disconnect)
        return arm,sdk
    def test_read_only_never_enables_moves_or_stops(self):
        arm,sdk=self.make_arm()
        self.assertTrue(arm.get_joint_state()[0])
        with self.assertRaises(RuntimeError): arm.enable()
        arm.disconnect()
        self.assertEqual(sdk.events,['connect','release'])
    def test_enable_primes_actual_position(self):
        arm,sdk=self.make_arm(enable_motion=True)
        arm.enable()
        self.assertEqual(sdk.events[1],('enable',sdk.q))
        arm.move_j_stream(sdk.q)
        arm.stop()
        self.assertFalse(arm.enabled)
    def test_wrong_firmware_rejected_without_motion(self):
        sdk=FakeArm(); sdk.version=lambda:100343
        with self.assertRaisesRegex(RuntimeError,'100341'):
            MarvinController('192.0.2.1',_backend=sdk)
        self.assertEqual(sdk.events,['connect','release'])
    def test_feedback_and_command_watchdogs_stop(self):
        for feedback in (True,False):
            arm,sdk=self.make_arm(enable_motion=True,feedback_timeout=.03,command_timeout=.05)
            arm.enable(); sdk.frozen=feedback
            deadline=time.monotonic()+.5
            while arm.enabled and time.monotonic()<deadline: time.sleep(.005)
            self.assertFalse(arm.enabled)
            self.assertIsNotNone(arm.fault)
            self.assertIn('stop',sdk.events)
            with self.assertRaises(RuntimeError): arm.enable()
    def test_excessive_step_rejected(self):
        arm,sdk=self.make_arm(enable_motion=True)
        arm.enable(); q=np.array(sdk.q);q[0]+=5
        with self.assertRaises(ValueError): arm.move_j_stream(q)
        self.assertFalse(any(isinstance(x,tuple) and x[0]=='command' for x in sdk.events))
    def test_version_buffer_and_success_code(self):
        native=_NativeSDK.__new__(_NativeSDK)
        def get(name,result):
            self.assertEqual(len(name),30)
            C.cast(result,C.POINTER(C.c_long)).contents.value=100341
            return 0
        native._get_int=get
        self.assertEqual(native.version(),100341)


class KinematicsTests(unittest.TestCase):
    def test_j67_coupling(self):
        model=MarvinModel()
        for j6,j7 in [(60,50),(-60,-50)]:
            q=[0,-30,0,-60,0,j6,j7]
            with self.assertRaises(ValueError):model.validate_degrees(q)
        model.validate_degrees([0,-30,0,-60,0,60,48])
    @unittest.skipUnless((VENDOR_ROOT/'SDK_PYTHON/libKine.so').exists(),'Build native SDK first')
    def test_native_fk_ik_roundtrip_and_tool_transform(self):
        _NativeSDK(VENDOR_ROOT,'A')
        k=MarvinKinematics(tool_xyz_m=(.01,.02,.03),tool_rpy_deg=(10,0,20))
        for degrees in ([21.8,-41,-4.74,-63.67,10.15,14.72,7.68], [10,-35,5,-65,10,5,2]):
            q=np.deg2rad(degrees);p,r=k.compute_fk(q)
            q2=k.solve_ik(q,p,r,initial_q=q)
            self.assertTrue(k.last_solve_info['success'],k.last_solve_info)
            np.testing.assert_allclose(q2,q,atol=1e-5)
            flange=np.array(k.sdk.fk(list(degrees)));flange[:3,3]/=1000
            np.testing.assert_allclose(p,(flange@k.tool)[:3,3],atol=1e-8)
        k.solve_ik(q,[20,20,20],r)
        self.assertFalse(k.last_solve_info['success'])


class LeftHandTests(unittest.TestCase):
    def test_left_model_and_readonly_hand(self):
        model,path,limits=hand_spec('left')
        self.assertIn('left',model);self.assertTrue(path.exists())
        hand=_FakeInitHand();hand.read_handedness=lambda:1
        sdk=SimpleNamespace(Hand=lambda **kwargs:hand)
        with patch.dict(sys.modules,{'wujihandpy':sdk}):
            backend=WujihandPyBackend(side='left',enable_motion=False,effort_limit=.6)
            self.assertEqual(backend.get_device_info()['handedness'],'left')
            np.testing.assert_allclose(backend.read_joint_positions(),hand.read_joint_actual_position().reshape(20))
            with self.assertRaises(RuntimeError):backend.send_joint_positions(np.zeros(20))
            backend.close()
        self.assertFalse(any(call.startswith('enabled=') for call in hand.calls))
        self.assertNotIn('write_effort_limit',hand.calls)
    def test_server_refuses_enable_without_permission(self):
        server=TeleopServer.__new__(TeleopServer);server.motion_enabled=False
        server.arm=SimpleNamespace(get_joint_state=lambda: (_ for _ in ()).throw(AssertionError('read')))
        server._enable_teleop()

    def test_hand_enable_is_deferred_and_uses_feedback(self):
        hand=_FakeInitHand();hand.read_handedness=lambda:1
        sdk=SimpleNamespace(Hand=lambda **kwargs:hand,
                            filter=SimpleNamespace(LowPass=lambda **kwargs:None))
        with patch.dict(sys.modules,{'wujihandpy':sdk}):
            backend=WujihandPyBackend(side='left',enable_motion=True,defer_enable=True)
            self.assertFalse(any(call.startswith('enabled=') for call in hand.calls))
            backend.enable()
            self.assertIn('enabled=True',hand.calls)
            np.testing.assert_allclose(backend.read_joint_positions(),hand.read_joint_actual_position().reshape(20))
            backend.close()


@unittest.skipUnless((VENDOR_ROOT/'SDK_PYTHON/libKine.so').exists(), 'Build native SDK first')
class ServerIntegrationTests(unittest.TestCase):
    def test_marvin_left_startup_and_pause_without_hardware(self):
        from utils.data_collector_wuji import DataCollectorWuji
        class Arm:
            version=100341
            def __init__(self,*args,**kwargs):self.commands=[];self.enabled=False
            def get_joint_state(self):return True,[21.8,-41,-4.74,-63.67,10.15,14.72,7.68]
            def enable(self):self.enabled=True
            def stop(self):self.enabled=False
            def move_j_stream(self,q):self.commands.append(q)
            def disconnect(self):self.stop()
        with tempfile.TemporaryDirectory() as td:
            def collector(**kwargs):
                kwargs['save_dir']=td
                return DataCollectorWuji(**kwargs)
            with patch('utils.marvin_controller.MarvinController',Arm), patch('scripts.teleop_real_server_wuji.DataCollectorWuji',collector):
                server=TeleopServer('tcp://127.0.0.1:*','tcp://127.0.0.1:*',
                    hand_backend='mock',hand_device=None,flange_thickness=0,camera_mode='none',
                    arm_backend='marvin',hand_side='left',enable_motion=False)
                try:
                    self.assertIsNone(server.keypoint_kinematics)
                    self.assertEqual(server.hand_model,'wujihand-left-v1.0.1')
                    server._process_msg({'type':'control','action':'teleop_on','hand_side':'left'})
                    self.assertFalse(server.teleop_enabled)
                    self.assertEqual(server.arm.commands,[])
                    server._sense_and_update(time.time())
                    server._process_msg({'type':'control','action':'teleop_off'})
                finally:
                    server.cleanup()


if __name__=='__main__':unittest.main()
