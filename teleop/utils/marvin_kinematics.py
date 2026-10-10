"""M6-CCS 4.0 left/right kinematics; public units are metres and radians."""
from pathlib import Path
import importlib.util
import hashlib
import ctypes as C
import numpy as np
from scipy.spatial.transform import Rotation

VENDOR_ROOT = Path(__file__).resolve().parents[2] / 'third_party' / 'marvin'
DEFAULT_CONFIG = VENDOR_ROOT / 'CommonConfig/config/ccs_m6_40.MvKDCfg'


def load_vendor_module(name, sdk_root=VENDOR_ROOT):
    path = Path(sdk_root).resolve() / 'SDK_PYTHON' / f'{name}.py'
    if not path.is_file():
        raise FileNotFoundError(f'Missing vendor SDK: {path}')
    spec = importlib.util.spec_from_file_location(f'marvin_vendor_{name}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MarvinModel:
    def __init__(self, path=DEFAULT_CONFIG, side='A'):
        if side not in ('A', 'B'):
            raise ValueError('MARVIN side must be A or B')
        self.path = Path(path).resolve()
        self.side = side
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        rows = [[float(x) for x in line.split(',') if x.strip()]
                for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
        if len(rows) != 26:
            raise ValueError('M6 config must contain both 13-line arm blocks')
        for offset in (0, 13):
            expected = [4] + [18]*7 + [4] + [3]*4
            if any(len(row) != size or not np.isfinite(row).all()
                   for row, size in zip(rows[offset:offset+13], expected)):
                raise ValueError('Malformed M6 configuration')
        block = rows[:13] if side == 'A' else rows[13:]
        if block[0][0] != 1017:
            raise ValueError('Expected CCS model type 1017')
        self.dh = np.array([row[:4] for row in block[1:9]])
        self.limits_deg = np.array([row[4:6] for row in block[1:8]])
        self.bd = np.array(block[9:13])
        if np.any(self.limits_deg[:, 0] >= self.limits_deg[:, 1]):
            raise ValueError('Invalid joint ranges')
        # This adapter is deliberately limited to the confirmed M6 4.0 profile.
        expected_dh = [[0,0,174.5,0],[90,0,0,0],[-90,0,287,0],
                       [90,18,0,180],[90,18,314,180],[90,0,0,90],
                       [90,0,0,90],[90,0,95,90]]
        expected_bd = [[0,-1.025,110.5],[0,1.025,110.5],
                       [0,-1.025,-110.5],[0,1.025,-110.5]]
        if not np.allclose(self.dh, expected_dh) or not np.allclose(self.bd, expected_bd):
            raise ValueError('Configuration is not the verified M6-CCS 4.0 profile')

    def validate_degrees(self, q, margin=0.0):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError('Expected seven finite MARVIN joint angles')
        lo, hi = self.limits_deg[:, 0].copy(), self.limits_deg[:, 1].copy()
        j6 = q[5]
        upper, lower = (self.bd[0], self.bd[3]) if j6 >= 0 else (self.bd[1], self.bd[2])
        hi[6] = min(hi[6], np.polyval(upper, j6))
        lo[6] = max(lo[6], np.polyval(lower, j6))
        if np.any(q < lo + margin) or np.any(q > hi - margin):
            raise ValueError('MARVIN joint limit or J6/J7 interference boundary exceeded')
        return q


class MarvinKinematics:
    def __init__(self, config_path=DEFAULT_CONFIG, side='A', sdk_root=VENDOR_ROOT,
                 tool_xyz_m=(0,0,0), tool_rpy_deg=(0,0,0)):
        self.model_config = MarvinModel(config_path, side)
        values = np.asarray([*tool_xyz_m, *tool_rpy_deg], dtype=float)
        if values.shape != (6,) or not np.isfinite(values).all():
            raise ValueError('Tool transform must be six finite values')
        self.tool = np.eye(4)
        self.tool[:3, 3] = values[:3]
        self.tool[:3, :3] = Rotation.from_euler('xyz', values[3:], degrees=True).as_matrix()
        self.module = load_vendor_module('fx_kine', sdk_root)
        self.sdk = self.module.Marvin_Kine()
        self.sdk.log_switch(0)
        abi = self.sdk.kine.TeleopABI
        abi.argtypes = [C.c_int]; abi.restype = C.c_size_t
        structure = self.module.FX_InvKineSolvePara
        expected = [C.sizeof(structure)] + [getattr(structure, field).offset for field in (
            'm_Input_IK_RefJoint', 'm_Output_RetJoint', 'm_Output_IsOutRange', 'm_Output_RunLmtP')]
        if any(abi(i) != v for i,v in enumerate(expected)):
            raise RuntimeError('MARVIN native/Python IK ABI mismatch')
        index = 0 if side == 'A' else 1
        cfg = self.sdk.load_config(arm_type=index, config_path=str(self.model_config.path))
        if not cfg or not self.sdk.initial_kine(cfg['TYPE'][index], cfg['DH'][index],
                                                cfg['PNVA'][index], cfg['BD'][index]):
            raise RuntimeError('MARVIN kinematics initialization failed')
        # Keep vendor FK at the native flange; apply the tool exactly once here.
        self.last_solve_info = {'success': False, 'status': 'not_started', 'iter_count': 0}

    def _tool(self, flange_thickness):
        if flange_thickness != 0:
            raise ValueError('MARVIN uses --tool-xyz-m/--tool-rpy-deg; flange-thickness must be 0')
        return self.tool

    def compute_fk(self, joint_angles, flange_thickness=0.0):
        q = self.model_config.validate_degrees(np.rad2deg(joint_angles))
        result = self.sdk.fk(q.tolist())
        if result is False or result is None:
            raise RuntimeError('MARVIN FK failed')
        transform = np.asarray(result, dtype=float)
        if transform.shape != (4,4) or not np.isfinite(transform).all():
            raise RuntimeError('MARVIN FK returned invalid pose')
        transform[:3, 3] *= 0.001
        transform = transform @ self._tool(flange_thickness)
        return transform[:3, 3].copy(), Rotation.from_matrix(transform[:3, :3]).as_quat()

    def solve_ik(self, current_q, target_pos, target_quat, flange_thickness=0.0, initial_q=None):
        self.last_solve_info = {'success': False, 'status': 'rejected', 'iter_count': 0}
        try:
            reference = self.model_config.validate_degrees(np.rad2deg(
                current_q if initial_q is None else initial_q))
            pos, quat = np.asarray(target_pos, float), np.asarray(target_quat, float)
            if pos.shape != (3,) or quat.shape != (4,) or not np.isfinite(np.r_[pos,quat]).all():
                raise ValueError('Invalid target pose')
            target = np.eye(4)
            target[:3, :3] = Rotation.from_quat(quat).as_matrix()
            target[:3, 3] = pos
            flange = target @ np.linalg.inv(self._tool(flange_thickness))
            flange[:3, 3] *= 1000.0
            params = self.module.FX_InvKineSolvePara()
            params.set_input_ik_target_tcp(flange.reshape(-1).tolist())
            params.set_input_ik_ref_joint(reference.tolist())
            params.set_input_ik_zsp_type(0)
            result = self.sdk.ik(params)
            if not result or result.m_Output_IsOutRange or result.m_Output_IsJntExd:
                raise ValueError('IK unreachable or out of bounds')
            if any(result.m_Output_IsDeg) or any(result.m_Output_JntExdTags):
                raise ValueError('IK singular or joint limit exceeded')
            q = self.model_config.validate_degrees(result.m_Output_RetJoint.to_list(), margin=0.5)
            if np.max(np.abs(q-reference)) > 10:
                raise ValueError('IK branch change exceeds 10 degrees')
            solved = np.deg2rad(q)
            fk_pos, fk_quat = self.compute_fk(solved)
            if np.linalg.norm(fk_pos-pos) > 0.002 or (
                Rotation.from_quat(fk_quat).inv()*Rotation.from_quat(quat)).magnitude() > np.deg2rad(1):
                raise ValueError('IK forward verification failed')
            self.last_solve_info.update(success=True, status='success')
            return solved
        except (ValueError, RuntimeError) as exc:
            self.last_solve_info['status'] = str(exc)
            return np.asarray(current_q, float).copy()
