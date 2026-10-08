import numpy as np
import pinocchio as pin
import casadi
import os
from scipy.spatial.transform import Rotation as R
from pinocchio import casadi as cpin


class RM75Kinematics:
    """
    Unified Kinematics Solver (IK & FK) for RM75 Robot.
    Based on Pinocchio and CasADi.
    """

    def __init__(self, urdf_path, base_pos=None, base_quat=None):
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        # 1. Load Model
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        # 2. Define End-Effector Frame
        self.ee_frame_name = "link_7"
        if self.model.existFrame(self.ee_frame_name):
            self.ee_frame_id = self.model.getFrameId(self.ee_frame_name)
        else:
            self.ee_frame_id = self.model.nframes - 1
            print(f"[Warn] Frame '{self.ee_frame_name}' not found, using last frame: {self.ee_frame_id}")

        # 3. Define Collision Frames (for IK)
        check_list = ["link_3", "link_4", "link_5", "link_6", "link_7"]
        self.collision_frame_ids = []
        for name in check_list:
            if self.model.existFrame(name):
                self.collision_frame_ids.append(self.model.getFrameId(name))

        # 4. Define World-to-Base Transform (T_wb)
        if base_pos is None: base_pos = [-0.4, 0.1, 0.0]
        if base_quat is None: base_quat = [0.0, 0.0, 0.0, 1.0]  # xyzw
        self.last_losses = {}
        self.last_solve_info = {
            "success": False,
            "status": "not_started",
            "iter_count": 0,
        }

        self.T_wb = np.eye(4)
        self.T_wb[:3, 3] = base_pos
        self.T_wb[:3, :3] = R.from_quat(base_quat).as_matrix()
        self.T_bw = np.linalg.inv(self.T_wb)  # Base-to-World (Inverse)

        # 5. Build IK Optimizer
        self._build_parametric_optimizer()
        print(f"[Init] RM75Kinematics initialized. EE Frame: {self.ee_frame_name}")

    def compute_fk(self, joint_angles, flange_thickness=0.0):
        """
        Forward Kinematics: Calculate TCP pose in World Frame given joint angles.

        Args:
            joint_angles (np.array): 7-DoF joint angles (radians).
            flange_thickness (float): Distance from flange (link_7) to TCP (tip) along Z-axis.

        Returns:
            tuple: (pos_world, quat_world_xyzw)
        """
        # 1. Update Pinocchio model
        pin.forwardKinematics(self.model, self.data, joint_angles)
        pin.updateFramePlacements(self.model, self.data)

        # 2. Get Flange Pose in Robot Base Frame (T_base_flange)
        T_bf = self.data.oMf[self.ee_frame_id].np  # 4x4 matrix

        # 3. Apply Flange/Tool Offset (T_flange_tcp)
        # Assuming the tool extends along the local Z-axis of the flange
        T_ft = np.eye(4)
        T_ft[2, 3] = flange_thickness

        # 4. Compute TCP in Base Frame: T_bt = T_bf * T_ft
        T_bt = T_bf @ T_ft

        # 5. Transform to World Frame: T_wt = T_wb * T_bt
        T_wt = self.T_wb @ T_bt

        # 6. Extract Position and Quaternion
        pos_world = T_wt[:3, 3]
        rot_world = T_wt[:3, :3]
        quat_world = R.from_matrix(rot_world).as_quat()  # xyzw

        return pos_world, quat_world

    def compute_fk_rot6d(self, joint_angles, flange_thickness=0.0):
        """
        Forward Kinematics: Calculate TCP pose in World Frame given joint angles.

        Args:
            joint_angles (np.array): 7-DoF joint angles (radians).
            flange_thickness (float): Distance from flange (link_7) to TCP (tip) along Z-axis.

        Returns:
            tuple: (pos_world, rot_6d)
                   pos_world is a 3D array [x, y, z].
                   rot_6d is a 6D array [R11, R21, R31, R12, R22, R32] representing the continuous rotation.
        """
        # 1. Update Pinocchio model
        pin.forwardKinematics(self.model, self.data, joint_angles)
        pin.updateFramePlacements(self.model, self.data)

        # 2. Get Flange Pose in Robot Base Frame (T_base_flange)
        T_bf = self.data.oMf[self.ee_frame_id].np  # 4x4 matrix

        # 3. Apply Flange/Tool Offset (T_flange_tcp)
        # Assuming the tool extends along the local Z-axis of the flange
        T_ft = np.eye(4)
        T_ft[2, 3] = flange_thickness

        # 4. Compute TCP in Base Frame: T_bt = T_bf @ T_ft
        T_bt = T_bf @ T_ft

        # 5. Transform to World Frame: T_wt = self.T_wb @ T_bt
        T_wt = self.T_wb @ T_bt

        # 6. Extract Position and 6D Continuous Rotation
        pos_world = T_wt[:3, 3]
        rot_world = T_wt[:3, :3]

        rot_6d = rot_world[:, :2].flatten().astype(float)

        return pos_world, rot_6d

    def solve_ik(
        self,
        current_q,
        target_pos_world,
        target_quat_world,
        flange_thickness=0.0,
        initial_q=None,
    ):
        """
        Inverse Kinematics: Calculate joint angles given target TCP pose in World Frame.
        (Renamed from 'solve' to 'solve_ik' for clarity, but logic is identical)
        """
        current_q = np.asarray(current_q, dtype=np.float64)
        if initial_q is None:
            initial_q = current_q
        initial_q = np.asarray(initial_q, dtype=np.float64)
        if current_q.shape != (self.model.nq,) or initial_q.shape != (self.model.nq,):
            raise ValueError(f"current_q and initial_q must have shape ({self.model.nq},)")
        initial_q = np.clip(
            initial_q,
            self.model.lowerPositionLimit,
            self.model.upperPositionLimit,
        )

        # Normalize Quaternion
        norm = np.linalg.norm(target_quat_world)
        if norm < 1e-6:
            target_quat_world = np.array([0, 0, 0, 1.0])
        else:
            target_quat_world = target_quat_world / norm

        # 1. Construct Target TCP Matrix in World
        rot_mat = R.from_quat(target_quat_world).as_matrix()
        T_tcp_world = np.eye(4)
        T_tcp_world[:3, :3] = rot_mat
        T_tcp_world[:3, 3] = target_pos_world

        # 2. Remove Flange Offset to get Flange Target (T_tcp_flange)
        # We need the pose of link_7, so we move backwards from TCP
        T_offset_inv = np.eye(4)
        T_offset_inv[2, 3] = -flange_thickness

        T_flange_world = T_tcp_world @ T_offset_inv

        t_pos_world_flange = T_flange_world[:3, 3]
        t_rot_world_flange = T_flange_world[:3, :3]

        # 3. Transform World Target to Local Base Target
        target_T_local = self._transform_target_to_local_numpy(t_pos_world_flange, t_rot_world_flange)

        t_pos_local = target_T_local[:3, 3]
        t_rot_local = target_T_local[:3, :3]

        # 4. Set Parameters for Optimizer
        self.opti.set_value(self.param_target_pos, t_pos_local)
        self.opti.set_value(self.param_target_rot, t_rot_local)
        self.opti.set_value(self.param_q_current, current_q)

        # Seed from the previous successful command while retaining measured
        # joints as the regularization reference.
        self.opti.set_initial(self.var_q, initial_q)

        # 5. Solve
        try:
            sol = self.opti.solve()
            q_opt = np.asarray(sol.value(self.var_q), dtype=np.float64)
            stats = self.opti.stats()
            status = str(stats.get("return_status", "unknown"))
            res = self.loss_evaluator(q_opt, t_pos_local, t_rot_local, current_q)
            self.last_losses = {
                "pos": float(res[0]),
                "rot": float(res[1]),
                "reg": float(res[2]),
                "posture": float(res[3]),
                "collision": float(res[4]),
                "total": float(res[5])
            }
            success = bool(
                status in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
                and np.all(np.isfinite(q_opt))
            )
            self.last_solve_info = {
                "success": success,
                "status": status,
                "iter_count": int(stats.get("iter_count", 0)),
                **self.last_losses,
            }
            return q_opt if success else current_q.copy()
        except Exception as exc:
            self.last_losses = {}
            try:
                stats = self.opti.stats()
            except Exception:
                stats = {}
            self.last_solve_info = {
                "success": False,
                "status": str(
                    stats.get("return_status", f"exception:{type(exc).__name__}")
                ),
                "iter_count": int(stats.get("iter_count", 0)),
            }
            return current_q.copy()

    def solve_ik_rot6d(self, current_q, target_pos_world, target_rot6d_world, flange_thickness=0.0):
        """
        Inverse Kinematics for 9D pose target [xyz + rot6d].

        Args:
            current_q (np.array): Current 7-DoF joint angles (radians).
            target_pos_world (np.array): Target position in world frame, shape (3,).
            target_rot6d_world (np.array): Target 6D rotation representation, shape (6,).
            flange_thickness (float): Distance from flange (link_7) to TCP along local Z.

        Returns:
            np.array: Solved joint angles.
        """
        target_rot_world = self.rot6d_to_rotmat_numpy(target_rot6d_world)
        target_quat_world = R.from_matrix(target_rot_world).as_quat()  # xyzw
        return self.solve_ik(
            current_q=current_q,
            target_pos_world=target_pos_world,
            target_quat_world=target_quat_world,
            flange_thickness=flange_thickness,
        )

    def _transform_target_to_local_numpy(self, pos_w, rot_w_mat):
        T_world = np.eye(4)
        T_world[:3, :3] = rot_w_mat
        T_world[:3, 3] = pos_w
        return self.T_bw @ T_world

    def _build_parametric_optimizer(self):
        """
        Builds the CasADi optimization problem (Graph).
        Same as original logic.
        """
        q_sx = casadi.SX.sym('q_sx', self.model.nq)
        tgt_pos_sx = casadi.SX.sym('tgt_pos_sx', 3)
        tgt_rot_sx = casadi.SX.sym('tgt_rot_sx', 3, 3)
        q_curr_sx = casadi.SX.sym('q_curr_sx', self.model.nq)

        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()

        cpin.framesForwardKinematics(cmodel, cdata, q_sx)
        c_ee_transform = cdata.oMf[self.ee_frame_id]

        # 1. Position Error
        error_pos = c_ee_transform.translation - tgt_pos_sx
        cost_pos = casadi.sumsqr(error_pos)

        # 2. Rotation Error
        # cost_rot = casadi.sumsqr(tgt_rot_sx - c_ee_transform.rotation)
        rot_trace = casadi.trace(casadi.mtimes(tgt_rot_sx.T, c_ee_transform.rotation))
        cost_rot = 3.0 - rot_trace

        # 3. Regularization (stay close to current q)
        cost_reg = casadi.sumsqr(q_sx - q_curr_sx)

        # 4. Singularity Avoidance
        q_nominal_rad = np.array([0.0, 1.04719755, 0.0, 1.04719755, 0.0, 1.04719755, 0.0])
        q_nominal = casadi.SX(q_nominal_rad)
        cost_posture = casadi.sumsqr(q_sx - q_nominal)

        # 5. Collision Avoidance (Ground)
        cost_collision = 0
        collision_margin = 0.02
        T_wb_rot = casadi.SX(self.T_wb[:3, :3])
        T_wb_pos = casadi.SX(self.T_wb[:3, 3])

        for fid in self.collision_frame_ids:
            pos_local = cdata.oMf[fid].translation
            # Transform local point to world to check Z height
            pos_world = T_wb_rot @ pos_local + T_wb_pos
            z_height = pos_world[2]
            # Penalty if z < margin
            violation = casadi.fmax(0, collision_margin - z_height)
            cost_collision += casadi.sumsqr(violation)

        # Weights
        W_POS = 100.0
        W_ROT = 20.0
        W_REG = 0.5
        W_POSTURE = 0.005
        W_COL = 1000.0
        total_cost_sx = W_POS * cost_pos + W_ROT * cost_rot + W_REG * cost_reg + W_POSTURE * cost_posture + W_COL * cost_collision

        # Create Function
        self.cost_func = casadi.Function(
            'cost_func',
            [q_sx, tgt_pos_sx, tgt_rot_sx, q_curr_sx],
            [total_cost_sx]
        )

        self.loss_evaluator = casadi.Function(
            'loss_evaluator',
            [q_sx, tgt_pos_sx, tgt_rot_sx, q_curr_sx],
            [cost_pos, cost_rot, cost_reg, cost_posture, cost_collision, total_cost_sx]
        )

        # Create Solver
        self.opti = casadi.Opti()

        # Variables
        self.var_q = self.opti.variable(self.model.nq)

        # Parameters
        self.param_target_pos = self.opti.parameter(3)
        self.param_target_rot = self.opti.parameter(3, 3)
        self.param_q_current = self.opti.parameter(self.model.nq)

        # Objective
        cost_mx = self.cost_func(
            self.var_q,
            self.param_target_pos,
            self.param_target_rot,
            self.param_q_current
        )
        self.opti.minimize(cost_mx)

        # Bounds
        self.opti.subject_to(self.opti.bounded(
            self.model.lowerPositionLimit,
            self.var_q,
            self.model.upperPositionLimit
        ))

        # Solver Config
        p_opts = {'expand': True, 'print_time': False, 'error_on_fail': False}
        s_opts = {
            'max_iter': 20,
            'tol': 1e-3,
            'acceptable_tol': 1e-3,  # 允许求解器提前退出
            'print_level': 0,
            'linear_solver': 'mumps',
            'warm_start_init_point': 'yes'
        }
        self.opti.solver("ipopt", p_opts, s_opts)

    @staticmethod
    def rot6d_to_rotmat_numpy(rot6d):
        rot6d = np.asarray(rot6d, dtype=float)
        if rot6d.shape != (6,):
            raise ValueError(f"rot6d must be shape (6,), got {rot6d.shape}")

        mat32 = rot6d.reshape(3, 2)
        x_raw = mat32[:, 0]
        y_raw = mat32[:, 1]

        x = x_raw / (np.linalg.norm(x_raw) + 1e-8)
        y = y_raw - np.dot(x, y_raw) * x
        y = y / (np.linalg.norm(y) + 1e-8)
        z = np.cross(x, y)

        return np.stack([x, y, z], axis=1)


    def quat_to_rot6d_numpy(quat, seq='xyzw'):
        if seq == 'wxyz':
            quat = np.array([quat[1], quat[2], quat[3], quat[0]])

        rot_matrix = R.from_quat(quat).as_matrix()

        rot_6d = np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]])

        return rot_6d
        
if __name__ == "__main__":
    import time
    from Robotic_Arm.rm_robot_interface import *

    ROBOT_IP = "192.168.1.18"
    ROBOT_PORT = 8080

    print("=" * 60)
    print(">>> Initializing Kinematics Solvers...")

    # 1. Custom Solver
    urdf = "./assets/RM75/urdf/RM75-6F/urdf/RM75-6F.urdf"
    solver = RM75Kinematics(urdf, base_pos=[0, 0, 0], base_quat=[0, 0, 0, 1])
    print("[Custom] Solver initialized.")

    # 2. Official Solver (Connect to Robot)
    print(f"[Official] Connecting to Robot at {ROBOT_IP}...")
    robot = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = robot.rm_create_robot_arm(ROBOT_IP, ROBOT_PORT)

    if handle.id == -1:
        print(f"\n[FATAL] Failed to connect to robot at {ROBOT_IP}.")
        exit(1)
    else:
        print(f"[Official] Connected. Robot ID: {handle.id}")

    # 3. Benchmark
    N_SAMPLES = 100
    print(f"\n>>> Running Benchmark on {N_SAMPLES} random poses...")

    test_joints_rad = np.random.uniform(-3.0, 3.0, (N_SAMPLES, 7))

    dt_fk_custom = []
    dt_fk_official = []
    dt_ik_custom = []
    pos_errors = []
    rot_errors = []

    for i in range(N_SAMPLES):
        q_rad = test_joints_rad[i]
        q_deg = np.rad2deg(q_rad).tolist()

        # --- A. Test FK ---

        # 1. Custom FK
        t0 = time.perf_counter()
        pos_custom, quat_custom = solver.compute_fk(q_rad, flange_thickness=0.0)
        dt_fk_custom.append((time.perf_counter() - t0) * 1000)

        # 2. Official FK
        t0 = time.perf_counter()
        result = robot.rm_algo_forward_kinematics(q_deg, 0)
        dt_fk_official.append((time.perf_counter() - t0) * 1000)

        if result is not None and len(result) == 7:
            pos_official = np.array(result[:3])
            quat_official = np.array([result[4], result[5], result[6], result[3]])

            p_err = np.linalg.norm(pos_custom - pos_official)

            dot = np.abs(np.dot(quat_custom, quat_official))
            if dot > 1.0: dot = 1.0
            r_err_deg = 2 * np.arccos(dot) * (180 / np.pi)

            pos_errors.append(p_err)
            rot_errors.append(r_err_deg)
        else:
            print(f"[Error] Official FK returned unexpected format at index {i}: {result}")

        # --- B. Test IK ---
        t0 = time.perf_counter()
        q_sol = solver.solve_ik(q_rad, pos_custom, quat_custom, flange_thickness=0.0)
        dt_ik_custom.append((time.perf_counter() - t0) * 1000)

    # 4. Disconnect
    robot.rm_delete_robot_arm()

    # 5. Report
    print("\n" + "=" * 60)
    print("                 BENCHMARK REPORT (Avg of 100)")
    print("=" * 60)

    print(f"1. Time Consumption (Lower is Better):")
    print(f"   [Custom FK]   : {np.mean(dt_fk_custom):.4f} ms")
    print(f"   [Official FK] : {np.mean(dt_fk_official):.4f} ms")
    print(f"   [Custom IK]   : {np.mean(dt_ik_custom):.4f} ms")

    if len(pos_errors) > 0:
        print(f"\n2. Accuracy (Custom vs Official):")
        print(f"   Position Err  : {np.mean(pos_errors) * 1000:.4f} mm")
        print(f"   Rotation Err  : {np.mean(rot_errors):.4f} deg")

        print("-" * 60)
        if np.mean(pos_errors) < 1e-4:
            print(">>> [PASS] Custom FK matches Official SDK.")
        else:
            print(">>> [WARN] Discrepancy found. Check URDF/Base definitions.")
    else:
        print("\n[Error] No valid comparisons made.")
