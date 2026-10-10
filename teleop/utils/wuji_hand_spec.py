"""Model-derived Wuji Hand joint specification used by teleoperation."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


HAND_MODEL = "wujihand-right-v1.0.1"
HAND_DOF = 20
URDF_PATH = Path(__file__).resolve().parents[1] / "assets" / "wujihand_urdf" / "urdf" / "right.urdf"

# The hardware/backend contract uses this explicit order: thumb, then index
# through little finger, with each finger ordered from palm to fingertip.
JOINT_NAMES = tuple(
    f"finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)


def hand_spec(side="right"):
    if side not in ("left", "right"):
        raise ValueError("Hand side must be left or right")
    path = URDF_PATH.with_name(f"{side}.urdf")
    model = f"wujihand-{side}-v1.0.1"
    return model, path, _load_joint_limits(path, model)


def _load_joint_limits(path=URDF_PATH, model=HAND_MODEL) -> np.ndarray:
    root = ET.parse(path).getroot()
    if root.attrib.get("name") != model:
        raise ValueError(f"Unexpected Wuji model in {URDF_PATH}: {root.attrib.get('name')!r}")

    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    limits = []
    for name in JOINT_NAMES:
        joint = joints.get(name)
        if joint is None or joint.attrib.get("type") != "revolute":
            raise ValueError(f"Missing revolute Wuji joint in URDF: {name}")
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"Missing limits for Wuji joint: {name}")
        limits.append((float(limit.attrib["lower"]), float(limit.attrib["upper"])))

    movable = [joint for joint in root.findall("joint") if joint.attrib.get("type") != "fixed"]
    if len(movable) != HAND_DOF:
        raise ValueError(f"Expected {HAND_DOF} Wuji joints, found {len(movable)} in {URDF_PATH}")
    return np.asarray(limits, dtype=np.float64)


JOINT_LIMITS = _load_joint_limits()
# HOME_Q = np.zeros(HAND_DOF, dtype=np.float64)
HOME_Q = np.array([
                0.2, 0.4, 0.3, 0.2,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0],dtype=np.float64,)


def validate_joint_vector(values, *, label: str = "Wuji joint vector") -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (HAND_DOF,):
        raise ValueError(f"{label} must have shape ({HAND_DOF},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    return vector
