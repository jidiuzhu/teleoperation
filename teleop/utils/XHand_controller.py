import numpy as np
import time
from xhand_controller import xhand_control


class XHandDriver:
    def __init__(self, port="/dev/ttyUSB0", baud=3000000):
        self._device = xhand_control.XHandControl()
        self._cmd = xhand_control.HandCommand_t()
        self._hand_id = 0
        self._last_joints = np.zeros(12)

        # Mapping from Finger ID to Name
        self.FINGERTIP_IDS = {
            2: "Thumb",
            5: "Index",
            7: "Middle",
            9: "Ring",
            11: "Little"
        }

        # Open serial port
        rsp = self._device.open_serial(port, baud)
        if rsp.error_code != 0:
            raise ConnectionError(f"Failed to open XHand serial: {rsp.error_message}")

        # Scan for hand ID
        ids = self._device.list_hands_id()
        if not ids:
            self._device.close_device()
            raise ConnectionError("No XHand detected.")
        self._hand_id = ids[0]
        print(f"[Init] XHand Connected ID: {self._hand_id}")

        # Initialize PID and control modes for all 12 motors
        for i in range(12):
            self._cmd.finger_command[i].id = i
            self._cmd.finger_command[i].kp = 100
            self._cmd.finger_command[i].ki = 0
            self._cmd.finger_command[i].kd = 0
            self._cmd.finger_command[i].tor_max = 300
            self._cmd.finger_command[i].mode = 3  # Position Control Mode

    def send_joints(self, joint_vals):
        """
        Send joint position commands to the hand.
        Args:
            joint_vals (list/array): 12 float values for joint angles.
        """
        self._last_joints = np.array(joint_vals)
        for i in range(12):
            self._cmd.finger_command[i].position = float(joint_vals[i])
        self._device.send_command(self._hand_id, self._cmd)

    def move_to(self, target_joints, duration=2.0):
        """
        Smoothly interpolate to target joints over a duration.
        """
        start_joints = self._last_joints.copy()
        steps = int(duration / 0.02)
        if steps == 0:
            steps = 1
        for i in range(steps):
            alpha = (i + 1) / steps
            curr = start_joints * (1 - alpha) + target_joints * alpha
            self.send_joints(curr)
            time.sleep(0.02)
        self.send_joints(target_joints)

    def get_full_state(self):
        """
        Low-level function to read the raw state struct from serial.
        """
        err, state = self._device.read_state(self._hand_id, True)
        if err.error_code != 0:
            print(f"[Error] Read state failed: {err.error_message}")
            return False, None
        return True, state

    def get_current_joints(self):
        """
        Get only current joint angles (12 dims).
        """
        success, state = self.get_full_state()
        if not success:
            return None

        try:
            current_pos = [state.finger_state[i].position for i in range(12)]
            return np.array(current_pos)
        except AttributeError:
            return np.zeros(12)

    def get_tactile_data(self):
        """
        Get only tactile sensor data.
        """
        success, state = self.get_full_state()
        if not success:
            return None

        tactile_data = {}
        try:
            sensor_idx = 0
            sorted_fingertips = sorted(self.FINGERTIP_IDS.keys())

            for fid in sorted_fingertips:
                finger_name = self.FINGERTIP_IDS[fid]

                if sensor_idx < len(state.sensor_data):
                    data = state.sensor_data[sensor_idx]
                    calc_force = [data.calc_force.fx, data.calc_force.fy, data.calc_force.fz]
                    raw_force = [[f.fx, f.fy, f.fz] for f in data.raw_force]

                    tactile_data[finger_name] = {
                        "calc": calc_force,
                        "raw": raw_force
                    }
                    sensor_idx += 1
                else:
                    tactile_data[finger_name] = {
                        "calc": [0.0, 0.0, 0.0],
                        "raw": []
                    }
        except Exception as e:
            print(f"[Warn] Parse tactile data error: {e}")
            return {}

        return tactile_data

    def get_composite_state(self):
        """
        [NEW METHOD]
        Fetch both joint angles and tactile data in a SINGLE serial transaction.
        This significantly improves data collection frequency.

        Returns:
            dict or None: {
                "joints": np.array([12 floats]),
                "tactile": {
                    "Thumb": {"calc": [fx, fy, fz], "raw": [[x,y,z], ...]},
                    ...
                }
            }
        """
        # 1. Single IO call
        success, state = self.get_full_state()
        if not success:
            return None

        # 2. Parse Joints
        try:
            current_pos = [state.finger_state[i].position for i in range(12)]
            joints = np.array(current_pos)
        except AttributeError:
            joints = np.zeros(12)

        # 3. Parse Tactile
        tactile_data = {}
        try:
            sensor_idx = 0
            sorted_fingertips = sorted(self.FINGERTIP_IDS.keys())

            for fid in sorted_fingertips:
                finger_name = self.FINGERTIP_IDS[fid]

                if sensor_idx < len(state.sensor_data):
                    data = state.sensor_data[sensor_idx]

                    # Calculated force (Fx, Fy, Fz)
                    calc_force = [
                        data.calc_force.fx,
                        data.calc_force.fy,
                        data.calc_force.fz
                    ]

                    # Raw force points (array of vectors)
                    raw_force = [
                        [f.fx, f.fy, f.fz] for f in data.raw_force
                    ]

                    tactile_data[finger_name] = {
                        "calc": calc_force,
                        "raw": raw_force
                    }
                    sensor_idx += 1
                else:
                    # Fill default empty data if sensor packet is missing
                    tactile_data[finger_name] = {
                        "calc": [0.0, 0.0, 0.0],
                        "raw": []
                    }
        except Exception as e:
            print(f"[Warn] Composite parse error: {e}")
            return None

        return {
            "joints": joints,
            "tactile": tactile_data
        }

    def get_diagnostics(self):
        success, state = self.get_full_state()
        if not success: return None

        temps = []
        errors = []

        for i in range(12):
            f_state = state.finger_state[i]
            temps.append(f_state.temperature)
            has_error = (f_state.commboard_err != 0 or
                         f_state.jonitboard_err != 0 or
                         f_state.tipboard_err != 0)
            errors.append(has_error)

            if has_error:
                print(f"[Alert] Finger {i} Error: Comm={f_state.commboard_err}, "
                      f"Joint={f_state.jonitboard_err}, Tip={f_state.tipboard_err}")

        return {"temperature": temps, "errors": errors}

    def reset_tactile_sensors(self):
        """
        Zero (tare) the tactile sensors.
        Ensure no external force is applied when calling this.
        """
        sensor_id_all = 17
        err = self._device.reset_sensor(self._hand_id, sensor_id_all)
        if err.error_code == 0:
            print("[System] Tactile sensors reset (zeroed) successfully.")
        else:
            print(f"[Error] Failed to reset sensors: {err.error_message}")

    def get_device_info(self):
        _, info = self._device.read_device_info(self._hand_id)
        _, hand_type = self._device.get_hand_type(self._hand_id)
        _, sn = self._device.get_serial_number(self._hand_id)

        return {
            "serial_number": sn,
            "hand_type": "Right/Left" if hand_type == 0 else str(hand_type),
            "hardware_id": info.hand_id
        }

    def close(self):
        self._device.close_device()
        print("[System] XHand device closed.")


if __name__ == "__main__":
    print(">>> [Init] Connecting to XHand...")

    try:
        # 1. Initialize
        hand = XHandDriver(port="/dev/ttyUSB0", baud=3000000)
    except Exception as e:
        print(f"[Fatal] Connection failed: {e}")
        exit(1)

    try:
        # 2. Get and print device metadata
        print(">>> [Info] Reading device info...")
        info = hand.get_device_info()
        print(f"    Serial Number : {info.get('serial_number', 'Unknown')}")
        print(f"    Hand Type     : {info.get('hand_type', 'Unknown')}")
        print(f"    Hardware ID   : {info.get('hardware_id', 'Unknown')}")

        # 3. Reset Sensors (Tare)
        print(">>> [Calib] Resetting tactile sensors...")
        print("    !!! Please ensure NO contact with fingertips at this moment !!!")
        time.sleep(1)
        hand.reset_tactile_sensors()
        print("    [Done] Sensors reset successfully.")
        time.sleep(1)

        # 4. Enter Monitoring Loop
        print(">>> [Monitor] Starting real-time monitoring (Press Ctrl+C to exit)")
        print("    (Try pressing the fingertips to observe value changes)")

        while True:
            # Use the new composite method for efficiency testing
            state = hand.get_composite_state()

            if state is None:
                print("[Warn] Data read failed, skipping this loop...")
                time.sleep(0.5)
                continue

            joints = state["joints"]
            tactile = state["tactile"]

            # Diagnostics usually called separately as it's not high-frequency critical
            diag = hand.get_diagnostics()

            print("\n" * 2)
            print(f"=== Real-time Monitor | Time: {time.strftime('%H:%M:%S')} ===")

            if diag:
                temps = diag['temperature']
                avg_temp = sum(temps) / len(temps) if temps else 0
                max_temp = max(temps) if temps else 0
                has_error = any(diag['errors'])
                status_str = "NORMAL" if not has_error else "ERROR DETECTED!"
                print(f"[Health] Status: {status_str} | Avg Temp: {avg_temp:.1f}°C | Max Temp: {max_temp:.1f}°C")

            print("[Tactile] Fingertip Force Data (Calculated [Fx, Fy, Fz]):")

            fingers_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']

            for fname in fingers_order:
                data = tactile.get(fname)
                if data:
                    calc = data['calc']
                    raw_len = len(data['raw'])
                    force_str = f"[{calc[0]:.2f}, {calc[1]:.2f}, {calc[2]:.2f}]"
                    raw_preview = "No Data"
                    if raw_len > 0:
                        r = data['raw'][0]
                        raw_preview = f"pt0:[{r[0]:.0f},{r[1]:.0f},{r[2]:.0f}]..."
                    print(f"    {fname:<8}: Force {force_str} | Raw Pts: {raw_len}")

            if joints is not None:
                print(f"[Joints]  Joint[0] Pos: {joints[0]:.4f} rad")

            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n>>> [Stop] Test interrupted by user.")
    except Exception as e:
        print(f"\n>>> [Error] Exception occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        hand.close()
        print(">>> [Exit] Device connection closed.")
