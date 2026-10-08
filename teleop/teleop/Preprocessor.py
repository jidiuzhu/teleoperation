import math
import numpy as np
import time
from teleop.constants_vuer import grd_yup2grd_zup, hand2inspire, hand2xhand, hand2assemble, vuer2linker_l_finger, vuer2linker_l_arm
from teleop.motion_utils import mat_update, fast_mat_inv

from teleop.constants_vuer import grd_yup2grd_zup,\
    vuer2linker_l_finger, vuer2linker_l_arm, vuer2linker_r_finger,\
    vuer2linker_1, vuer2linker_2, vuer2linker_3, vuer2linker_4, vuer2linker_5, vuer2linker_6, vuer2linker_7, vuer2linker_8, vuer2linker_9


class VuerPreprocessor:
    def __init__(self):
        self.vuer_head_mat = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 1.5],
                                  [0, 0, 1, -0.2],
                                  [0, 0, 0, 1]])
        self.vuer_right_wrist_mat = np.array([[1, 0, 0, 0.5],
                                         [0, 1, 0, 1],
                                         [0, 0, 1, -0.5],
                                         [0, 0, 0, 1]])
        self.vuer_left_wrist_mat = np.array([[1, 0, 0, -0.5],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, -0.5],
                                        [0, 0, 0, 1]])

    def process(self, tv, hand="inspire"):
        if hand == "inspire":
            matrix = hand2inspire
        elif hand == "xhand":
            matrix = hand2xhand
        elif hand == "assemble_charger":
            matrix = hand2assemble
        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())

        # change of basis
        head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)

        rel_left_wrist_mat = left_wrist_mat @ matrix
        rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        rel_right_wrist_mat = right_wrist_mat @ matrix  # wTr = wTh @ hTr
        rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]

        # homogeneous
        left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
        right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (matrix.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (matrix.T @ rel_right_fingers)[0:3, :].T
        # 将所有要返回的变量放入一个字典，方便遍历和打印
        variables_to_inspect = {
            "head_mat": head_mat,
            "rel_left_wrist_mat": rel_left_wrist_mat,
            "rel_right_wrist_mat": rel_right_wrist_mat,
            "rel_left_fingers": rel_left_fingers,
            "rel_right_fingers": rel_right_fingers
        }

        # for name, var in variables_to_inspect.items():
        #     检查变量是否为 NumPy 数组，以安全地获取其形状
            # if isinstance(var, np.ndarray):
            #     print(f"Variable: {name:<22} | Type: {type(var).__name__:<15} | Shape: {var.shape}")
            # else:
            #     print(f"Variable: {name:<22} | Type: {type(var).__name__:<15} | Shape: (Not a NumPy array)")
        # print(rel_right_fingers)
        return head_mat, rel_left_wrist_mat, rel_right_wrist_mat, rel_left_fingers, rel_right_fingers

    def get_hand_gesture(self, tv):
        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())

        # change of basis
        right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
        left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)

        left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
        right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])

        # change of basis
        left_fingers = grd_yup2grd_zup @ left_fingers
        right_fingers = grd_yup2grd_zup @ right_fingers

        rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
        rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
        rel_left_fingers = (hand2inspire.T @ rel_left_fingers)[0:3, :].T
        rel_right_fingers = (hand2inspire.T @ rel_right_fingers)[0:3, :].T
        all_fingers = np.concatenate([rel_left_fingers, rel_right_fingers], axis=0)

        return all_fingers


# class VuerPreprocessorLegacy:
#     def __init__(self):
#         self.vuer_head_mat = np.eye(4)
#         self.vuer_right_wrist_mat = np.eye(4)
#         self.vuer_left_wrist_mat = np.eye(4)
#         self.rel_left_wrist_mat = np.zeros((4, 4), dtype=np.float32)
#         self.rel_right_wrist_mat = np.zeros((4, 4), dtype=np.float32)
#         self.rel_left_fingers_mat = np.zeros((25, 3), dtype=np.float32)
#         self.rel_right_fingers_mat = np.zeros((25, 3), dtype=np.float32)
#
#         self.urdf_y90_offset = np.array([
#             [0, 0, 1, 0],
#             [0, 1, 0, 0],
#             [-1, 0, 0, 0],
#             [0, 0, 0, 1]
#         ], dtype=np.float32)
#         self.vuer_head_mat = np.array([[1, 0, 0, 0],
#                                   [0, 1, 0, 1.5],
#                                   [0, 0, 1, -0.2],
#                                   [0, 0, 0, 1]])
#         self.vuer_right_wrist_mat = np.array([[1, 0, 0, 0.5],
#                                          [0, 1, 0, 1],
#                                          [0, 0, 1, -0.5],
#                                          [0, 0, 0, 1]])
#         self.vuer_left_wrist_mat = np.array([[1, 0, 0, -0.5],
#                                         [0, 1, 0, 1],
#                                         [0, 0, 1, -0.5],
#                                         [0, 0, 0, 1]])
#
#     def process_linker(self, tv):
#         self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
#
#         self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())
#         self.rel_left_wrist_mat[:3, 3] = (vuer2linker_l_arm[:3, :3]).T @ (
#                 self.vuer_left_wrist_mat[0:3, 3] - self.vuer_head_mat[0:3, 3])
#         self.rel_left_wrist_mat[:3, :3] = (fast_mat_inv(
#             vuer2linker_l_arm) @ self.vuer_left_wrist_mat @ vuer2linker_l_arm)[:3, :3]
#
#         self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
#         self.rel_right_wrist_mat[:3, 3] = (vuer2linker_l_arm[:3, :3]).T @ (
#                 self.vuer_right_wrist_mat[0:3, 3] - self.vuer_head_mat[0:3, 3])
#         self.rel_right_wrist_mat[:3, :3] = (fast_mat_inv(
#             vuer2linker_l_arm) @ self.vuer_right_wrist_mat @ vuer2linker_l_arm)[:3, :3]
#
#         self.rel_left_wrist_mat = self.rel_left_wrist_mat @ self.urdf_y90_offset
#         self.rel_right_wrist_mat = self.rel_right_wrist_mat @ self.urdf_y90_offset
#
#         vuer_left_fingers_homo = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
#         rel_left_fingers_homo = fast_mat_inv(self.vuer_left_wrist_mat) @ vuer_left_fingers_homo
#         self.rel_left_fingers_mat = (vuer2linker_l_finger.T @ rel_left_fingers_homo)[0:3, :].T
#
#         vuer_right_fingers_homo = np.concatenate(
#             [tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])
#         rel_right_fingers_homo = fast_mat_inv(self.vuer_right_wrist_mat) @ vuer_right_fingers_homo
#         self.rel_right_fingers_mat = (vuer2linker_l_finger.T @ rel_right_fingers_homo)[0:3, :].T
#
#         return self.rel_left_wrist_mat, self.rel_right_wrist_mat, self.rel_left_fingers_mat, self.rel_right_fingers_mat
#
#     def process(self, tv, hand="inspire"):
#         if hand == "inspire":
#             wrist_matrix = hand2inspire
#             finger_matrix = hand2inspire
#         elif hand == "xhand":
#             wrist_matrix = hand2xhand
#             finger_matrix = hand2xhand
#         elif hand == "assemble_charger":
#             wrist_matrix = hand2assemble
#             finger_matrix = hand2assemble
#         elif hand == "linker":
#             wrist_matrix = vuer2linker_l_arm
#             finger_matrix = vuer2linker_l_finger
#         else:
#             raise ValueError(f"Unknown hand type: {hand}")
#
#         self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())
#         self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
#         self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())
#
#         head_mat = grd_yup2grd_zup @ self.vuer_head_mat @ fast_mat_inv(grd_yup2grd_zup)
#         right_wrist_mat = grd_yup2grd_zup @ self.vuer_right_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
#         left_wrist_mat = grd_yup2grd_zup @ self.vuer_left_wrist_mat @ fast_mat_inv(grd_yup2grd_zup)
#
#         rel_left_wrist_mat = left_wrist_mat @ wrist_matrix
#         rel_left_wrist_mat[0:3, 3] = rel_left_wrist_mat[0:3, 3] - head_mat[0:3, 3]
#
#         rel_right_wrist_mat = right_wrist_mat @ wrist_matrix
#         rel_right_wrist_mat[0:3, 3] = rel_right_wrist_mat[0:3, 3] - head_mat[0:3, 3]
#
#         left_fingers = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
#         right_fingers = np.concatenate([tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])
#
#         left_fingers = grd_yup2grd_zup @ left_fingers
#         right_fingers = grd_yup2grd_zup @ right_fingers
#
#         rel_left_fingers = fast_mat_inv(left_wrist_mat) @ left_fingers
#         rel_right_fingers = fast_mat_inv(right_wrist_mat) @ right_fingers
#
#         rel_left_fingers = (finger_matrix.T @ rel_left_fingers)[0:3, :].T
#         rel_right_fingers = (finger_matrix.T @ rel_right_fingers)[0:3, :].T
#
#         return rel_left_wrist_mat, rel_right_wrist_mat, rel_left_fingers, rel_right_fingers


class VuerPreprocessorLegacy:
    def __init__(self):
        self.vuer_head_mat = np.eye(4)
        self.vuer_right_wrist_mat = np.eye(4)
        self.vuer_left_wrist_mat = np.eye(4)
        self.rel_left_wrist_mat = np.zeros((4, 4), dtype=np.float32)
        self.rel_right_wrist_mat = np.zeros((4, 4), dtype=np.float32)
        self.rel_left_fingers_mat = np.zeros((25, 3), dtype=np.float32)
        self.rel_right_fingers_mat = np.zeros((25, 3), dtype=np.float32)

    def process_linker(self, tv):
        # 绕 z 轴旋转 -90°
        vuer2linker_r_arm_right1 = np.array([[0, 1, 0, 0],
                                             [-1, 0, 0, 0],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]])

        # 绕 x 轴旋转 180°
        vuer2linker_r_arm_right2 = np.array([[1, 0, 0, 0],
                                             [0, -1, 0, 0],
                                             [0, 0, -1, 0],
                                             [0, 0, 0, 1]])

        # 绕 y 轴旋转 -90度
        vuer2linker_r_arm_left1 = np.array([[0, 0, -1, 0],
                                            [0, 1, 0, 0],
                                            [1, 0, 0, 0],
                                            [0, 0, 0, 1]])

        # 绕 z 轴旋转 -90°
        vuer2linker_r_arm_left2 = np.array([[0, 1, 0, 0],
                                            [-1, 0, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]])

        self.vuer_head_mat = mat_update(self.vuer_head_mat, tv.head_matrix.copy())

        self.vuer_left_wrist_mat = mat_update(self.vuer_left_wrist_mat, tv.left_hand.copy())
        self.rel_left_wrist_mat[:3, 3] = (vuer2linker_6 @ vuer2linker_1)[:3, :3] @ (
                    self.vuer_left_wrist_mat[0:3, 3] - self.vuer_head_mat[0:3, 3])
        self.rel_left_wrist_mat[:3, :3] = (
                                                      vuer2linker_6 @ vuer2linker_1 @ self.vuer_left_wrist_mat @ vuer2linker_5 @ vuer2linker_8)[
                                          :3, :3]

        self.vuer_right_wrist_mat = mat_update(self.vuer_right_wrist_mat, tv.right_hand.copy())
        self.rel_right_wrist_mat[:3, 3] = (vuer2linker_4 @ vuer2linker_6)[:3, :3] @ (
                    self.vuer_right_wrist_mat[0:3, 3] - self.vuer_head_mat[0:3, 3])
        self.rel_right_wrist_mat[:3, :3] = (
                                                       vuer2linker_4 @ vuer2linker_6 @ self.vuer_right_wrist_mat @ vuer2linker_6 @ vuer2linker_7)[
                                           :3, :3]

        vuer_left_fingers_homo = np.concatenate([tv.left_landmarks.copy().T, np.ones((1, tv.left_landmarks.shape[0]))])
        rel_left_fingers_homo = fast_mat_inv(self.vuer_left_wrist_mat) @ vuer_left_fingers_homo
        self.rel_left_fingers_mat = (vuer2linker_l_finger.T @ rel_left_fingers_homo)[0:3, :].T

        vuer_right_fingers_homo = np.concatenate(
            [tv.right_landmarks.copy().T, np.ones((1, tv.right_landmarks.shape[0]))])
        rel_right_fingers_homo = fast_mat_inv(self.vuer_right_wrist_mat) @ vuer_right_fingers_homo
        self.rel_right_fingers_mat = (vuer2linker_r_finger.T @ rel_right_fingers_homo)[0:3, :].T

        return self.rel_left_wrist_mat, self.rel_right_wrist_mat, self.rel_left_fingers_mat, self.rel_right_fingers_mat
