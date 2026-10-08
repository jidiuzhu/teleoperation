import time
import threading
import numpy as np
from Robotic_Arm.rm_robot_interface import *


class RoboticArmController:
    def __init__(self, ip="192.168.1.18", port=8080, stream_hz=120):
        self.ip = ip
        self.port = port
        self.connected = False

        self.robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.robot.rm_create_robot_arm(self.ip, self.port)

        if self.handle.id == -1:
            raise ConnectionError(f"Failed to connect to Robotic Arm at {self.ip}:{self.port}")

        self.connected = True
        print(f"[Init] Robot Arm Connected at {self.ip}")

        self.stream_hz = stream_hz
        self.dt = 1.0 / self.stream_hz

        self.target_q = None
        self.current_q_cmd = None

        self.streaming = False
        self.stream_thread = None
        self.lock = threading.Lock()

    def start_high_freq_stream(self, initial_q):
        if not self.connected:
            return

        self.target_q = np.array(initial_q, dtype=np.float64)
        self.current_q_cmd = np.array(initial_q, dtype=np.float64)
        self.streaming = True

        self.stream_thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.stream_thread.start()
        print(f"[System] High-frequency stream thread started at {self.stream_hz}Hz.")

    def stop_high_freq_stream(self):
        self.streaming = False
        if self.stream_thread is not None:
            self.stream_thread.join(timeout=1.0)
            print("[System] High-frequency stream thread stopped.")

    def _stream_loop(self):
        alpha = 0.08

        interval = self.dt
        next_time = time.perf_counter() + interval

        while self.streaming and self.connected:
            with self.lock:
                tgt = self.target_q.copy()
                curr = self.current_q_cmd.copy()

            next_q = curr * (1 - alpha) + tgt * alpha

            with self.lock:
                self.current_q_cmd = next_q

            ret = self.robot.rm_movej_canfd(next_q.tolist(), True)

            now = time.perf_counter()
            sleep_time = next_time - now

            if sleep_time > 0.002:
                time.sleep(sleep_time - 0.002)

            while time.perf_counter() < next_time:
                pass

            next_time += interval

    def move_j_blocking(self, joint_angles, speed=20):
        if not self.connected: return
        if self.streaming:
            print("[Warn] Streaming！")
            return

        ret = self.robot.rm_movej(joint_angles, speed, 0, 0, 1)
        if ret != 0:
            print(f"[Warn] move_j_blocking returned code: {ret}")

    def move_j_stream(self, joint_angles):
        if not self.connected: return

        if self.streaming:
            with self.lock:
                self.target_q = np.array(joint_angles, dtype=np.float64)
        else:
            ret = self.robot.rm_movej_canfd(joint_angles, False)

    def get_joint_state(self):
        if not self.connected:
            return False, []
        code, state = self.robot.rm_get_current_arm_state()
        if code == 0:
            return True, state['joint']
        else:
            return False, []

    def disconnect(self):
        if self.connected:
            self.stop_high_freq_stream()
            self.robot.rm_delete_robot_arm()
            self.connected = False
            print("[System] Robot Arm disconnected.")
            

    def __del__(self):
        self.disconnect()
