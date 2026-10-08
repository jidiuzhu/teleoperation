"""Wuji-specific DexPilot configuration and morphology correction."""

from pathlib import Path

import numpy as np
import yaml

from utils.wuji_hand_spec import (
    HAND_DOF,
    JOINT_NAMES,
    validate_joint_vector,
)


TELEOP_ROOT = Path(__file__).resolve().parents[1]

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
FINGER_MCP_INDICES = np.asarray((1, 5, 10, 15, 20), dtype=np.int32)
FINGER_INTERNAL_INDICES = np.asarray((3, 8, 13, 18, 23), dtype=np.int32)
FINGER_TIP_INDICES = np.asarray((4, 9, 14, 19, 24), dtype=np.int32)
MIN_FINGERTIP_SCALE = 0.5
MAX_FINGERTIP_SCALE = 2.0
DEFAULT_INTERNAL_LINK_NAMES = tuple(
    f"finger{finger}_link3" for finger in range(1, 6)
)
DEFAULT_WRIST_TO_JOINT_WEIGHT = 1.0
DEFAULT_JOINT_TO_TIP_WEIGHT = 2.0
MAX_INTERNAL_CONSTRAINT_WEIGHT = 50.0


def load_fingertip_scales(config):
    """Load ordered per-finger scales from the optional Wuji config section."""
    morphology = config.get("wuji_morphology", {})
    if morphology is None:
        morphology = {}
    if not isinstance(morphology, dict):
        raise ValueError("wuji_morphology must be a mapping")

    unknown_morphology = set(morphology) - {"fingertip_scale"}
    if unknown_morphology:
        raise ValueError(
            "Unknown wuji_morphology fields: "
            f"{sorted(unknown_morphology)}"
        )

    values = morphology.get("fingertip_scale", {})
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError("wuji_morphology.fingertip_scale must be a mapping")

    unknown_fingers = set(values) - set(FINGER_NAMES)
    if unknown_fingers:
        raise ValueError(
            "Unknown Wuji fingertip_scale names: "
            f"{sorted(unknown_fingers)}; expected {FINGER_NAMES}"
        )

    scales = []
    for name in FINGER_NAMES:
        raw_value = values.get(name, 1.0)
        if isinstance(raw_value, bool):
            raise ValueError(f"Wuji fingertip scale for {name} must be numeric")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Wuji fingertip scale for {name} must be numeric"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(f"Wuji fingertip scale for {name} must be finite")
        if not MIN_FINGERTIP_SCALE <= value <= MAX_FINGERTIP_SCALE:
            raise ValueError(
                f"Wuji fingertip scale for {name} must be within "
                f"[{MIN_FINGERTIP_SCALE}, {MAX_FINGERTIP_SCALE}], got {value}"
            )
        scales.append(value)

    return np.asarray(scales, dtype=np.float64)


def scale_fingertips_from_mcp(landmarks, scales):
    """Scale each MCP-to-tip vector while leaving all other landmarks intact."""
    points = np.asarray(landmarks, dtype=np.float64)
    scale_values = np.asarray(scales, dtype=np.float64)
    if points.shape != (25, 3):
        raise ValueError(
            f"Vuer right-hand landmarks must have shape (25, 3), got {points.shape}"
        )
    if scale_values.shape != (len(FINGER_NAMES),):
        raise ValueError(
            f"Wuji fingertip scales must have shape ({len(FINGER_NAMES)},), "
            f"got {scale_values.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("Vuer right-hand landmarks contain non-finite values")
    if not np.all(np.isfinite(scale_values)):
        raise ValueError("Wuji fingertip scales contain non-finite values")

    corrected = points.copy()
    mcp_points = points[FINGER_MCP_INDICES]
    tip_vectors = points[FINGER_TIP_INDICES] - mcp_points
    corrected[FINGER_TIP_INDICES] = mcp_points + scale_values[:, None] * tip_vectors
    return corrected


def scale_finger_targets_from_mcp(landmarks, scales):
    """Scale each internal landmark and fingertip from the same MCP anchor."""
    corrected = scale_fingertips_from_mcp(landmarks, scales)
    points = np.asarray(landmarks, dtype=np.float64)
    scale_values = np.asarray(scales, dtype=np.float64)
    mcp_points = points[FINGER_MCP_INDICES]
    internal_vectors = points[FINGER_INTERNAL_INDICES] - mcp_points
    corrected[FINGER_INTERNAL_INDICES] = (
        mcp_points + scale_values[:, None] * internal_vectors
    )
    return corrected


def _load_finger_weights(section, field, default):
    values = section.get(field, {})
    if values is None:
        values = {}
    if not isinstance(values, dict):
        raise ValueError(f"wuji_internal_constraints.{field} must be a mapping")
    unknown_fingers = set(values) - set(FINGER_NAMES)
    if unknown_fingers:
        raise ValueError(
            f"Unknown {field} finger names: {sorted(unknown_fingers)}; "
            f"expected {FINGER_NAMES}"
        )

    weights = []
    for finger_name in FINGER_NAMES:
        raw_value = values.get(finger_name, default)
        if isinstance(raw_value, bool):
            raise ValueError(f"{field} for {finger_name} must be numeric")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} for {finger_name} must be numeric") from exc
        if not np.isfinite(value):
            raise ValueError(f"{field} for {finger_name} must be finite")
        if not 0.0 <= value <= MAX_INTERNAL_CONSTRAINT_WEIGHT:
            raise ValueError(
                f"{field} for {finger_name} must be within "
                f"[0, {MAX_INTERNAL_CONSTRAINT_WEIGHT}], got {value}"
            )
        weights.append(value)
    return np.asarray(weights, dtype=np.float32)


def load_internal_constraint_config(config):
    """Load Wuji internal-link names and per-finger objective weights."""
    section = config.get("wuji_internal_constraints", {})
    if section is None:
        section = {}
    if not isinstance(section, dict):
        raise ValueError("wuji_internal_constraints must be a mapping")

    allowed_fields = {
        "enabled",
        "internal_link_names",
        "wrist_to_joint_weight",
        "joint_to_tip_weight",
    }
    unknown_fields = set(section) - allowed_fields
    if unknown_fields:
        raise ValueError(
            "Unknown wuji_internal_constraints fields: "
            f"{sorted(unknown_fields)}"
        )

    enabled = section.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("wuji_internal_constraints.enabled must be boolean")

    internal_link_names = section.get(
        "internal_link_names", list(DEFAULT_INTERNAL_LINK_NAMES)
    )
    if not isinstance(internal_link_names, (list, tuple)):
        raise ValueError(
            "wuji_internal_constraints.internal_link_names must be a list"
        )
    internal_link_names = tuple(internal_link_names)
    if len(internal_link_names) != len(FINGER_NAMES):
        raise ValueError(
            "wuji_internal_constraints.internal_link_names must contain "
            f"{len(FINGER_NAMES)} links"
        )
    if any(not isinstance(name, str) or not name for name in internal_link_names):
        raise ValueError("Wuji internal link names must be non-empty strings")
    if len(set(internal_link_names)) != len(internal_link_names):
        raise ValueError("Wuji internal link names must be unique")

    return {
        "enabled": enabled,
        "internal_link_names": internal_link_names,
        "wrist_to_joint_weights": _load_finger_weights(
            section, "wrist_to_joint_weight", DEFAULT_WRIST_TO_JOINT_WEIGHT
        ),
        "joint_to_tip_weights": _load_finger_weights(
            section, "joint_to_tip_weight", DEFAULT_JOINT_TO_TIP_WEIGHT
        ),
    }


def format_fingertip_scales(scales):
    values = np.asarray(scales, dtype=np.float64)
    return ", ".join(
        f"{name}={value:.3f}" for name, value in zip(FINGER_NAMES, values)
    )


class WujiDexPilotRetargeter:
    """Build DexPilot once and expose hardware-ordered Wuji targets."""

    def __init__(self, config_path):
        try:
            from dex_retargeting.retargeting_config import RetargetingConfig
        except ImportError as exc:
            raise RuntimeError(
                "dex_retargeting is not installed in this Python environment"
            ) from exc

        config_path = Path(config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Wuji retargeting config not found: {config_path}")

        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        if not isinstance(config, dict) or "right" not in config:
            raise ValueError(f"Missing right-hand config in {config_path}")
        if not isinstance(config["right"], dict):
            raise ValueError(f"right-hand config must be a mapping in {config_path}")

        self.fingertip_scales = load_fingertip_scales(config)
        self.internal_constraints = load_internal_constraint_config(config)

        RetargetingConfig.set_default_urdf_dir(str(TELEOP_ROOT / "assets"))
        retargeting_config = RetargetingConfig.from_dict(config["right"])
        self._retargeting = retargeting_config.build()

        if self.internal_constraints["enabled"]:
            from dex_retargeting.seq_retarget import SeqRetargeting

            from utils.wuji_dexpilot_optimizer import WujiDexPilotOptimizer

            base_optimizer = self._retargeting.optimizer
            optimizer = WujiDexPilotOptimizer(
                robot=base_optimizer.robot,
                target_joint_names=list(base_optimizer.target_joint_names),
                finger_tip_link_names=list(retargeting_config.finger_tip_link_names),
                wrist_link_name=retargeting_config.wrist_link_name,
                target_link_human_indices=np.asarray(
                    base_optimizer.target_link_human_indices, dtype=np.int32
                ),
                internal_link_names=list(
                    self.internal_constraints["internal_link_names"]
                ),
                internal_human_indices=FINGER_INTERNAL_INDICES,
                fingertip_human_indices=FINGER_TIP_INDICES,
                wrist_to_joint_weights=self.internal_constraints[
                    "wrist_to_joint_weights"
                ],
                joint_to_tip_weights=self.internal_constraints[
                    "joint_to_tip_weights"
                ],
                huber_delta=base_optimizer.huber_loss.beta,
                norm_delta=base_optimizer.norm_delta,
                project_dist=base_optimizer.project_dist,
                escape_dist=base_optimizer.escape_dist,
                eta1=base_optimizer.eta1,
                eta2=base_optimizer.eta2,
                scaling=base_optimizer.scaling,
            )
            if base_optimizer.adaptor is not None:
                optimizer.set_kinematic_adaptor(base_optimizer.adaptor)
            self._retargeting = SeqRetargeting(
                optimizer,
                has_joint_limits=retargeting_config.has_joint_limits,
                lp_filter=self._retargeting.filter,
            )

        indices = np.asarray(
            self._retargeting.optimizer.target_link_human_indices,
            dtype=np.int32,
        )
        if indices.ndim != 2 or indices.shape[0] != 2:
            raise ValueError(
                "DexPilot target_link_human_indices must have shape (2, N), "
                f"got {indices.shape}"
            )
        if np.any(indices < 0) or np.any(indices >= 25):
            raise ValueError("DexPilot human landmark indices must be within [0, 24]")
        self._origin_indices = indices[0]
        self._task_indices = indices[1]

        self.retarget_joint_names = tuple(self._retargeting.joint_names)
        missing = [
            name for name in JOINT_NAMES if name not in self.retarget_joint_names
        ]
        extra = [name for name in self.retarget_joint_names if name not in JOINT_NAMES]
        if missing or extra or len(self.retarget_joint_names) != HAND_DOF:
            raise RuntimeError(
                "Wuji retargeting joints do not match the hardware model: "
                f"missing={missing}, extra={extra}, got={self.retarget_joint_names}"
            )
        self._hardware_order = np.asarray(
            [self.retarget_joint_names.index(name) for name in JOINT_NAMES],
            dtype=np.int32,
        )

    @property
    def fingertip_scale_summary(self):
        return format_fingertip_scales(self.fingertip_scales)

    @property
    def internal_constraint_summary(self):
        if not self.internal_constraints["enabled"]:
            return "disabled"
        wrist_weights = self.internal_constraints["wrist_to_joint_weights"]
        tip_weights = self.internal_constraints["joint_to_tip_weights"]
        return ", ".join(
            f"{name}={wrist:.1f}/{tip:.1f}"
            for name, wrist, tip in zip(FINGER_NAMES, wrist_weights, tip_weights)
        )

    def build_reference(self, right_hand_landmarks):
        corrected = scale_finger_targets_from_mcp(
            right_hand_landmarks,
            self.fingertip_scales,
        )
        return corrected[self._task_indices] - corrected[self._origin_indices]

    def retarget(self, right_hand_landmarks):
        reference = self.build_reference(right_hand_landmarks)
        retargeted = np.asarray(
            self._retargeting.retarget(reference), dtype=np.float64
        )
        expected_shape = (len(self.retarget_joint_names),)
        if retargeted.shape != expected_shape:
            raise ValueError(
                "DexPilot output must have shape "
                f"{expected_shape}, got {retargeted.shape}"
            )
        return validate_joint_vector(
            retargeted[self._hardware_order], label="Wuji DexPilot target"
        )


__all__ = [
    "DEFAULT_INTERNAL_LINK_NAMES",
    "FINGER_INTERNAL_INDICES",
    "FINGER_MCP_INDICES",
    "FINGER_NAMES",
    "FINGER_TIP_INDICES",
    "WujiDexPilotRetargeter",
    "format_fingertip_scales",
    "load_fingertip_scales",
    "load_internal_constraint_config",
    "scale_finger_targets_from_mcp",
    "scale_fingertips_from_mcp",
]
