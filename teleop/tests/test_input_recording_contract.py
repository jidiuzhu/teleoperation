from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock

import h5py
import numpy as np

TELEOP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TELEOP_ROOT))

from scripts.teleop_real_server_wuji import TeleopServer
from utils.data_collector_wuji import DataCollectorWuji
from utils.wuji_hand_spec import JOINT_NAMES


class InputRecordingContractTest(unittest.TestCase):
    def test_glove_origin_is_saved_in_common_hdf5_schema(self):
        with TemporaryDirectory() as directory:
            collector = DataCollectorWuji(
                save_dir=directory,
                hand_model="wujihand-right-v1.0.1",
                hand_joint_names=JOINT_NAMES,
                hand_urdf="right.urdf",
                flange_thickness=0.026,
                camera_mode="none",
                camera_names=(),
                distance_keypoint_names=(),
            )
            collector.start_recording("demo", "glove")
            collector.stop_recording()
            with h5py.File(collector.filepath, "r") as recording:
                self.assertEqual(recording.attrs["input_device"], "glove")
                self.assertTrue(recording.attrs["recording_complete"])

    def test_server_stops_when_commands_expire(self):
        server = TeleopServer.__new__(TeleopServer)
        server.teleop_enabled = True
        server.last_command_time = time.monotonic() - 1.0
        server.human_anchor_pos = np.ones(3)
        server.human_anchor_rot = np.eye(3)
        server._finish_recording = Mock()

        server._expire_teleop_if_stale()

        self.assertFalse(server.teleop_enabled)
        self.assertIsNone(server.human_anchor_pos)
        server._finish_recording.assert_called_once()

    def test_server_forwards_input_source_and_rejects_bad_motion(self):
        server = TeleopServer.__new__(TeleopServer)
        server.collector = Mock()
        server.teleop_enabled = True
        server.last_command_time = time.monotonic()
        server._update_arm_target_from_wrist = Mock()
        server.hand_target_q = np.zeros(20)
        server._process_msg({"type": "control", "action": "start_recording", "task_name": "demo", "input_device": "glove"})
        server.collector.start_recording.assert_called_once_with("demo", "glove")

        previous_time = server.last_command_time
        server._process_msg({"type": "cmd", "right_wrist_mat": [float("nan")] * 16, "hand_qpos": [0.0] * 20})
        self.assertEqual(server.last_command_time, previous_time)
        server._update_arm_target_from_wrist.assert_not_called()

        reflection = np.eye(4)
        reflection[0, 0] = -1
        server._process_msg({"type": "cmd", "right_wrist_mat": reflection.ravel().tolist(), "hand_qpos": [0.0] * 20})
        server._update_arm_target_from_wrist.assert_not_called()

        server._process_msg({"type": "cmd", "right_wrist_mat": np.eye(4).ravel().tolist(), "hand_qpos": [0.0] * 20})
        server._update_arm_target_from_wrist.assert_called_once()
        self.assertGreater(server.last_command_time, previous_time)


if __name__ == "__main__":
    unittest.main()
