"""DexPilot optimizer with Wuji finger-direction constraints."""

from typing import List

import numpy as np
import torch

from dex_retargeting.optimizer import DexPilotOptimizer
from dex_retargeting.robot_wrapper import RobotWrapper


class WujiDexPilotOptimizer(DexPilotOptimizer):
    """Preserve DexPilot fingertip objectives and add internal-link vectors."""

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        target_link_human_indices: np.ndarray,
        internal_link_names: List[str],
        internal_human_indices: np.ndarray,
        fingertip_human_indices: np.ndarray,
        wrist_to_joint_weights: np.ndarray,
        joint_to_tip_weights: np.ndarray,
        **kwargs,
    ):
        super().__init__(
            robot=robot,
            target_joint_names=target_joint_names,
            finger_tip_link_names=finger_tip_link_names,
            wrist_link_name=wrist_link_name,
            target_link_human_indices=target_link_human_indices,
            **kwargs,
        )

        expected_fingers = self.num_fingers
        if len(internal_link_names) != expected_fingers:
            raise ValueError(
                f"Expected {expected_fingers} Wuji internal links, "
                f"got {len(internal_link_names)}"
            )
        if len(set(internal_link_names)) != expected_fingers:
            raise ValueError("Wuji internal link names must be unique")
        if len(finger_tip_link_names) != expected_fingers:
            raise ValueError("Wuji fingertip and internal link counts must match")

        internal_human_indices = np.asarray(internal_human_indices, dtype=np.int32)
        fingertip_human_indices = np.asarray(fingertip_human_indices, dtype=np.int32)
        wrist_to_joint_weights = np.asarray(
            wrist_to_joint_weights, dtype=np.float32
        )
        joint_to_tip_weights = np.asarray(joint_to_tip_weights, dtype=np.float32)
        expected_shape = (expected_fingers,)
        named_values = {
            "internal_human_indices": internal_human_indices,
            "fingertip_human_indices": fingertip_human_indices,
            "wrist_to_joint_weights": wrist_to_joint_weights,
            "joint_to_tip_weights": joint_to_tip_weights,
        }
        for name, values in named_values.items():
            if values.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}, got {values.shape}"
                )
        if np.any(internal_human_indices < 0) or np.any(
            internal_human_indices >= 25
        ):
            raise ValueError("Wuji internal human indices must be within [0, 24]")
        if np.any(fingertip_human_indices < 0) or np.any(
            fingertip_human_indices >= 25
        ):
            raise ValueError("Wuji fingertip human indices must be within [0, 24]")
        for name, weights in (
            ("wrist_to_joint_weights", wrist_to_joint_weights),
            ("joint_to_tip_weights", joint_to_tip_weights),
        ):
            if not np.all(np.isfinite(weights)) or np.any(weights < 0):
                raise ValueError(f"{name} must contain finite non-negative values")

        base_indices = np.asarray(self.target_link_human_indices, dtype=np.int32)
        expected_base_vectors = expected_fingers * (expected_fingers - 1) // 2
        expected_base_vectors += expected_fingers
        if base_indices.shape != (2, expected_base_vectors):
            raise ValueError(
                "Base DexPilot human indices must have shape "
                f"(2, {expected_base_vectors}), got {base_indices.shape}"
            )

        wrist_human_indices = np.zeros(expected_fingers, dtype=np.int32)
        extra_human_indices = np.stack(
            [
                np.concatenate([wrist_human_indices, internal_human_indices]),
                np.concatenate([internal_human_indices, fingertip_human_indices]),
            ],
            axis=0,
        )
        self.target_link_human_indices = np.concatenate(
            [base_indices, extra_human_indices], axis=1
        )

        self.base_vector_count = expected_base_vectors
        self.internal_vector_count = expected_fingers * 2
        self.internal_constraint_weights = np.concatenate(
            [wrist_to_joint_weights, joint_to_tip_weights]
        )

        self.origin_link_names.extend(
            [wrist_link_name] * expected_fingers + list(internal_link_names)
        )
        self.task_link_names.extend(
            list(internal_link_names) + list(finger_tip_link_names)
        )

        # Rebuild the kinematics cache because the standard DexPilot cache only
        # contains the wrist and fingertip links.
        self.computed_link_names = list(
            dict.fromkeys(self.origin_link_names + self.task_link_names)
        )
        self.origin_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in self.origin_link_names]
        )
        self.task_link_indices = torch.tensor(
            [self.computed_link_names.index(name) for name in self.task_link_names]
        )
        self.computed_link_indices = self.get_link_indices(self.computed_link_names)

    def get_objective_function(
        self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray
    ):
        target_vector = np.asarray(target_vector)
        expected_shape = (
            self.base_vector_count + self.internal_vector_count,
            3,
        )
        if target_vector.shape != expected_shape:
            raise ValueError(
                f"Wuji DexPilot target vectors must have shape {expected_shape}, "
                f"got {target_vector.shape}"
            )

        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj],
            target_vec_dist[len_s1:len_proj] <= 0.03,
        )

        normal_weight = np.ones(len_proj, dtype=np.float32)
        high_weight = np.asarray(
            [25] * len_s1 + [50] * len_s2, dtype=np.float32
        )
        pair_weight = np.where(self.projected, high_weight, normal_weight)
        wrist_weight = np.ones(self.num_fingers, dtype=np.float32) * 3
        # wrist_weight *= len_proj + self.num_fingers
        base_weight = torch.from_numpy(
            np.concatenate([pair_weight, wrist_weight])
        )
        internal_weight = torch.from_numpy(self.internal_constraint_weights)

        normal_vec = target_vector * self.scaling
        direction_vec = target_vector[:len_proj] / (
            target_vec_dist[:, None] + 1e-6
        )
        projected_vec = direction_vec * self.projected_dist[:, None]
        reference_vec = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )
        reference_vec = np.concatenate(
            [reference_vec, normal_vec[len_proj:]], axis=0
        )
        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [
                self.robot.get_link_pose(index)
                for index in self.computed_link_indices
            ]
            body_pos = np.asarray([pose[:3, 3] for pose in target_link_poses])
            torch_body_pos = torch.as_tensor(body_pos)
            torch_body_pos.requires_grad_()

            origin_link_pos = torch_body_pos[self.origin_link_indices, :]
            task_link_pos = torch_body_pos[self.task_link_indices, :]
            robot_vec = task_link_pos - origin_link_pos
            vector_distance = torch.norm(
                robot_vec - torch_target_vec, dim=1, keepdim=False
            )
            per_vector_loss = self.huber_loss(
                vector_distance, torch.zeros_like(vector_distance)
            )

            # Keep the original 15-vector DexPilot loss normalization exactly;
            # the ten direction constraints are an additive term.
            base_loss = (
                per_vector_loss[: self.base_vector_count]
                * base_weight
                / self.base_vector_count
            ).sum()
            internal_loss = (
                per_vector_loss[self.base_vector_count :]
                * internal_weight
                / self.base_vector_count
            ).sum()
            total_loss = base_loss + internal_loss
            result = total_loss.cpu().detach().item()

            if grad.size > 0:
                jacobians = []
                for index, link_pose in zip(
                    self.computed_link_indices, target_link_poses
                ):
                    link_body_jacobian = (
                        self.robot.compute_single_link_local_jacobian(qpos, index)[
                            :3, ...
                        ]
                    )
                    link_rotation = link_pose[:3, :3]
                    jacobians.append(link_rotation @ link_body_jacobian)

                jacobians = np.stack(jacobians, axis=0)
                total_loss.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]
                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, np.asarray(jacobians))
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)
                grad[:] = grad_qpos[:]

            return result

        return objective


__all__ = ["WujiDexPilotOptimizer"]
