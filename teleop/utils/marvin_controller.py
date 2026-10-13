"""Serialized MARVIN 100341 control. Connection is read-only by default."""
import ctypes as C
import threading
import time
import numpy as np
from utils.marvin_kinematics import MarvinModel, VENDOR_ROOT, DEFAULT_CONFIG, load_vendor_module


class MarvinController:
    def __init__(self, ip, *, side='A', sdk_root=VENDOR_ROOT, config_path=DEFAULT_CONFIG,
                 enable_motion=False, velocity_ratio=5, acceleration_ratio=5,
                 max_joint_speed_deg_s=20.0, feedback_timeout=0.25, command_timeout=0.3,
                 _backend=None):
        self.model = MarvinModel(config_path, side)
        if not 1 <= velocity_ratio <= 10 or not 1 <= acceleration_ratio <= 10:
            raise ValueError('Initial MARVIN velocity/acceleration ratios must be 1..10')
        if not np.isfinite([max_joint_speed_deg_s, feedback_timeout, command_timeout]).all() or min(max_joint_speed_deg_s, feedback_timeout, command_timeout) <= 0:
            raise ValueError('Invalid controller limits')
        self.side, self.index = side, 0 if side == 'A' else 1
        self.motion_allowed = enable_motion
        self.vel, self.acc = velocity_ratio, acceleration_ratio
        self.max_speed = max_joint_speed_deg_s
        self.feedback_timeout, self.command_timeout = feedback_timeout, command_timeout
        self.lock = threading.RLock()
        self.connected = self.enabled = False
        self.last_frame = self.last_feedback_time = self.last_command_time = None
        self.last_command = None
        self.fault = None
        self.quit_event = threading.Event()
        self.thread = None
        self.sdk = _backend or _NativeSDK(sdk_root, side)
        try:
            if not self.sdk.connect(ip):
                raise ConnectionError('MARVIN connection failed')
            self.connected = True
            version = self.sdk.version()
            if not str(version).startswith('100341'):
                raise RuntimeError(f'Expected 100341 controller, got {version}')
            self.version = version
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                with self.lock:
                    self._read()
                if self.last_feedback_time is not None:
                    break
                time.sleep(0.02)
            if self.last_feedback_time is None:
                raise TimeoutError('MARVIN feedback frame counter is not advancing')
            self.thread = threading.Thread(target=self._watchdog, daemon=True)
            self.thread.start()
        except BaseException:
            self.disconnect()
            raise

    def _read(self):
        data = self.sdk.read()
        state, output = data['states'][self.index], data['outputs'][self.index]
        if state['err_code'] or state['cur_state'] == 100:
            raise RuntimeError(f'MARVIN fault: {state}')
        q = self.model.validate_degrees(output['fb_joint_pos'])
        frame = output['frame_serial']
        # Only advancing firmware feedback, not repeated reads, refreshes age.
        if self.last_frame is not None and frame != self.last_frame:
            self.last_feedback_time = time.monotonic()
        self.last_frame = frame
        self.current_state, self.current_q = state['cur_state'], q
        return q

    def _fresh(self):
        if self.last_feedback_time is None or time.monotonic()-self.last_feedback_time > self.feedback_timeout:
            raise TimeoutError('MARVIN feedback is stale')

    def get_joint_state(self):
        with self.lock:
            if not self.connected or self.fault:
                return False, []
            try:
                q = self._read()
                self._fresh()
                return True, q.tolist()
            except Exception as exc:
                if self.enabled:
                    self._trip(exc)
                return False, []

    def enable(self):
        with self.lock:
            if not self.motion_allowed:
                raise RuntimeError('Motion not authorized: use --enable-motion')
            if self.fault:
                raise RuntimeError(f'MARVIN requires restart after fault: {self.fault}')
            q = self._read(); self._fresh()
            if self.current_state != 0:
                raise RuntimeError('MARVIN must be idle before enabling; use vendor controls')
            # Prime current feedback as target before entering position mode.
            self.enabled = True  # cleanup must stop even if acknowledgement fails
            try:
                self.sdk.configure(self.vel, self.acc, q.tolist())
                deadline = time.monotonic()+2
                while time.monotonic() < deadline:
                    self._read(); self._fresh()
                    if self.current_state == 1:
                        self.last_command = self.current_q.copy()
                        self.last_command_time = time.monotonic()
                        return
                    time.sleep(0.01)
                raise TimeoutError('MARVIN did not enter position mode')
            except Exception as exc:
                self._trip(exc)
                raise

    def limit_joint_target(self, joint_angles):
        """Shape a teleop target to the existing speed bound; never send here."""
        with self.lock:
            if not self.enabled or self.fault or self.last_command_time is None:
                raise RuntimeError('MARVIN is not enabled')
            target = self.model.validate_degrees(joint_angles, margin=0.5)
            dt = max(0.0, min(time.monotonic()-self.last_command_time, 0.1))
            delta = target-self.last_command
            largest = float(np.max(np.abs(delta)))
            allowed = self.max_speed*dt
            if largest > allowed:
                target = self.last_command + delta*(allowed/largest)
            # Interpolation must also satisfy the coupled wrist boundaries.
            return self.model.validate_degrees(target, margin=0.5).copy()

    def move_j_stream(self, joint_angles):
        with self.lock:
            if not self.enabled or self.fault:
                raise RuntimeError('MARVIN is not enabled')
            q = self._read(); self._fresh()
            if self.current_state != 1:
                raise RuntimeError('MARVIN left position mode')
            target = self.model.validate_degrees(joint_angles, margin=0.5)
            if np.max(np.abs(q-self.last_command)) > 10:
                self._trip('tracking error exceeds 10 degrees')
                raise RuntimeError(self.fault)
            dt = min(time.monotonic()-self.last_command_time, 0.1)
            if np.max(np.abs(target-self.last_command)) > self.max_speed*dt + 0.1:
                raise ValueError('MARVIN command exceeds configured joint slew limit')
            if not self.sdk.command(target.tolist()):
                self._trip('command submission failed')
                raise RuntimeError(self.fault)
            self.last_command, self.last_command_time = target.copy(), time.monotonic()

    def stop(self):
        with self.lock:
            if self.enabled:
                self.sdk.stop()
                self.enabled = False
            self.last_command_time = None

    def _trip(self, reason):
        self.fault = str(reason)
        try:
            self.stop()
        except Exception:
            self.enabled = False

    def _watchdog(self):
        while not self.quit_event.wait(0.02):
            with self.lock:
                if not self.enabled:
                    continue
                try:
                    self._read(); self._fresh()
                    if self.current_state != 1:
                        raise RuntimeError('Unexpected MARVIN mode')
                    if self.last_command_time is None or time.monotonic()-self.last_command_time > self.command_timeout:
                        raise TimeoutError('MARVIN command watchdog expired')
                except Exception as exc:
                    self._trip(exc)

    def disconnect(self):
        self.quit_event.set()
        with self.lock:
            try:
                self.stop()
            finally:
                if self.connected:
                    self.sdk.release()
                    self.connected = False
        if self.thread is not None:
            self.thread.join(timeout=1)


class _NativeSDK:
    def __init__(self, root, side):
        module = load_vendor_module('fx_robot', root)
        self.state = module.DCSS()
        self.to_dict = module.structure2dict
        self.side = side
        path = str(root) + '/SDK_PYTHON/libMarvinSDK.so'
        self.lib = C.CDLL(path)
        abi = self.lib.TeleopABI
        abi.argtypes = [C.c_int]; abi.restype = C.c_size_t
        expected = [C.sizeof(module.DCSS), C.sizeof(module.StateCtr), C.sizeof(module.RT_IN),
                    C.sizeof(module.RT_OUT), module.DCSS.m_State.offset, module.DCSS.m_In.offset,
                    module.DCSS.m_Out.offset, module.DCSS.m_ParaValueI.offset,
                    module.DCSS.m_ParaRetSerial.offset, C.sizeof(C.c_long)]
        if any(abi(i) != v for i,v in enumerate(expected)):
            raise RuntimeError('MARVIN native/Python feedback ABI mismatch')
        def bind(name, args, result):
            fn = getattr(self.lib, name); fn.argtypes = args; fn.restype = result
            return fn
        self._connect = bind('OnLinkTo', [C.c_ubyte]*4, C.c_bool)
        self._read = bind('OnGetBuf', [C.POINTER(module.DCSS)], C.c_bool)
        self._get_int = bind('OnGetIntPara', [C.c_char_p,C.POINTER(C.c_long)], C.c_long)
        self._clear = bind('OnClearSet', [], C.c_bool)
        self._send = bind('OnSetSend', [], C.c_bool)
        self._target = bind('OnSetJointCmdPos_'+side,[C.POINTER(C.c_double)],C.c_bool)
        self._mode = bind('OnSetTargetState_'+side,[C.c_int],C.c_bool)
        self._limits = bind('OnSetJointLmt_'+side,[C.c_int,C.c_int],C.c_bool)
        self._stop = bind('OnEMG_'+side,[],None)
        self._release = bind('OnRelease',[],None)

    def connect(self, ip):
        import ipaddress
        return self._connect(*ipaddress.IPv4Address(ip).packed)

    def version(self):
        result = C.c_long()
        status = self._get_int(C.create_string_buffer(b'VERSION', 30), C.byref(result))
        if status != 0:
            raise RuntimeError('Cannot read controller VERSION')
        return result.value

    def read(self):
        if not self._read(C.byref(self.state)):
            raise ConnectionError('MARVIN feedback read failed')
        return self.to_dict(self.state)

    def _begin_transaction(self, timeout_s=0.1):
        # OnSetSend only marks the native buffer pending (m_SendTag=100).
        # DoSend clears that flag asynchronously after sending. A false clear
        # therefore means busy, not a failed motion command. Retry only clear;
        # never stage fields until it succeeds and never resubmit a command.
        deadline = time.monotonic() + timeout_s
        while True:
            if self._clear():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('MARVIN transmit buffer remained busy for 100 ms')
            time.sleep(min(0.001, remaining))

    def configure(self, vel, acc, q):
        # The 100341 API stages settings in shared native state.
        self._begin_transaction()
        ok = self._limits(vel,acc) and self._target((C.c_double*7)(*q))
        if not ok or not self._send():
            raise RuntimeError('Failed to initialize MARVIN target/limits')
        self._begin_transaction()
        if not self._mode(1) or not self._send():
            raise RuntimeError('Failed to request position mode')

    def command(self, q):
        self._begin_transaction()
        return self._target((C.c_double*7)(*q)) and self._send()

    def stop(self): self._stop()
    def release(self): self._release()
