import time
import unittest

import numpy as np
from scipy.spatial.transform import Rotation as R

from scripts.teleop_real_server_wuji import (
    MAX_EE_DISPLACEMENT_M,
    TeleopServer,
    limit_pose_step,
)


class WujiServerPoseMappingTest(unittest.TestCase):
    def test_pose_step_limits_translation_and_rotation(self):
        current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        desired_rotation = R.from_euler("z", 90.0, degrees=True).as_quat()
        desired = np.concatenate([[1.0, 0.0, 0.0], desired_rotation])

        limited = limit_pose_step(
            current, desired, max_translation=0.02, max_rotation_rad=np.deg2rad(5.0)
        )

        self.assertAlmostEqual(np.linalg.norm(limited[:3]), 0.02)
        angle = (R.from_quat(limited[3:]) * R.from_quat(current[3:]).inv()).magnitude()
        self.assertAlmostEqual(angle, np.deg2rad(5.0))

    def test_wrist_rotation_is_relative_to_activation_pose(self):
        server = TeleopServer.__new__(TeleopServer)
        server.fixed_rot_mat = np.eye(3)
        server.robot_anchor_pos = np.array([0.4, 0.0, 0.3])
        server.robot_anchor_rot = R.from_euler("y", 90.0, degrees=True).as_matrix()
        server.human_anchor_pos = None
        server.human_anchor_rot = None
        anchor_quat = R.from_matrix(server.robot_anchor_rot).as_quat()
        server.desired_ee_pose = np.concatenate(
            [server.robot_anchor_pos, anchor_quat]
        )

        human_anchor = np.eye(4)
        human_anchor[:3, :3] = R.from_euler(
            "xyz", [20.0, -10.0, 30.0], degrees=True
        ).as_matrix()
        human_anchor[:3, 3] = [0.1, 0.2, 0.3]
        server._update_arm_target_from_wrist(human_anchor.reshape(-1).tolist())

        human_delta = R.from_euler("z", 15.0, degrees=True).as_matrix()
        human_current = human_anchor.copy()
        human_current[:3, :3] = human_delta @ human_anchor[:3, :3]
        human_current[:3, 3] += [0.05, -0.02, 0.01]
        server._update_arm_target_from_wrist(human_current.reshape(-1).tolist())

        expected_rotation = human_delta @ server.robot_anchor_rot
        actual_rotation = R.from_quat(server.desired_ee_pose[3:]).as_matrix()
        np.testing.assert_allclose(actual_rotation, expected_rotation, atol=1e-8)
        np.testing.assert_allclose(
            server.desired_ee_pose[:3],
            server.robot_anchor_pos + np.array([0.05, -0.02, 0.01]),
        )

    def test_wrist_displacement_is_bounded(self):
        server = TeleopServer.__new__(TeleopServer)
        server.fixed_rot_mat = np.eye(3)
        server.robot_anchor_pos = np.array([0.4, 0.0, 0.3])
        server.robot_anchor_rot = np.eye(3)
        server.human_anchor_pos = np.zeros(3)
        server.human_anchor_rot = np.eye(3)

        wrist = np.eye(4)
        wrist[:3, 3] = [10.0, 0.0, 0.0]
        server._update_arm_target_from_wrist(wrist.reshape(-1).tolist())

        displacement = server.desired_ee_pose[:3] - server.robot_anchor_pos
        self.assertAlmostEqual(np.linalg.norm(displacement), MAX_EE_DISPLACEMENT_M)


class _RejectingKinematics:
    def __init__(self):
        self.last_solve_info = {}

    def solve_ik(self, current_q, *args, **kwargs):
        self.last_solve_info = {
            "success": False,
            "status": "Maximum_CpuTime_Exceeded",
            "iter_count": 1,
        }
        return np.asarray(current_q).copy()


class _CountingArm:
    def __init__(self):
        self.commands = []

    def move_j_stream(self, command):
        self.commands.append(np.asarray(command))


class _CountingHand:
    def __init__(self):
        self.commands = []

    def send_joints(self, command):
        self.commands.append(np.asarray(command))


class WujiServerIkFailureTest(unittest.TestCase):
    def test_failed_ik_is_not_sent_to_arm(self):
        server = TeleopServer.__new__(TeleopServer)
        server.teleop_enabled = True
        server.teleop_start_time = time.time() - 10.0
        server.arm_target_q_rad = np.zeros(7)
        server.desired_ee_pose = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
        server.last_valid_ee_pose = server.desired_ee_pose.copy()
        server.flange_thickness = 0.026
        server.kinematics = _RejectingKinematics()
        server.arm = _CountingArm()
        server.hand = _CountingHand()
        server.hand_command_q = np.zeros(20)
        server.hand_target_q = np.zeros(20)
        server.hand_limits_min = np.full(20, -1.0)
        server.hand_limits_max = np.full(20, 1.0)
        observation = {
            "arm_qpos": np.zeros(7),
            "ee_pose": server.last_valid_ee_pose.copy(),
        }

        costs = server._step_control(time.time(), observation)

        self.assertEqual(costs["ik_failed"], 1)
        self.assertEqual(server.arm.commands, [])
        self.assertEqual(len(server.hand.commands), 1)


if __name__ == "__main__":
    unittest.main()
