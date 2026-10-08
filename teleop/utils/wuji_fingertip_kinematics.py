"""Forward kinematics for Wuji fingertips mounted on an RM75 arm."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


_TELEOP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARM_URDF = (
    _TELEOP_ROOT / "assets" / "RM75" / "urdf" / "RM75-6F" / "urdf" / "RM75-6F.urdf"
)
DEFAULT_HAND_URDF = (
    _TELEOP_ROOT / "assets" / "wujihand_urdf" / "urdf" / "right.urdf"
)

ARM_JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
HAND_JOINT_NAMES = tuple(
    f"finger{finger}_joint{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
)
FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
FINGERTIP_LINK_NAMES = tuple(
    f"finger{finger}_tip_link" for finger in range(1, 6)
)


def _parse_vector(text, *, default):
    if text is None:
        return np.asarray(default, dtype=np.float64)
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid URDF 3-vector: {text!r}")
    return values


def _rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _origin_transform(origin):
    transform = np.eye(4, dtype=np.float64)
    if origin is None:
        return transform
    transform[:3, 3] = _parse_vector(origin.attrib.get("xyz"), default=(0, 0, 0))
    transform[:3, :3] = _rpy_matrix(
        _parse_vector(origin.attrib.get("rpy"), default=(0, 0, 0))
    )
    return transform


def _axis_angle_transform(axis, angle):
    norm = np.linalg.norm(axis)
    if norm <= 0.0:
        raise ValueError("Revolute URDF joint has a zero-length axis")
    x, y, z = axis / norm
    c = np.cos(angle)
    s = np.sin(angle)
    one_minus_c = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    return transform


@dataclass(frozen=True)
class _Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray


class _UrdfTree:
    def __init__(self, urdf_path):
        self.path = Path(urdf_path)
        if not self.path.is_file():
            raise FileNotFoundError(f"URDF not found: {self.path}")

        root = ET.parse(self.path).getroot()
        self.link_names = tuple(link.attrib["name"] for link in root.findall("link"))
        links = set(self.link_names)
        self._joint_by_child = {}
        self.movable_joint_names = []
        self.edges = []

        for element in root.findall("joint"):
            name = element.attrib["name"]
            joint_type = element.attrib["type"]
            if joint_type not in {"fixed", "revolute", "continuous"}:
                raise ValueError(
                    f"Unsupported joint type {joint_type!r} for {name!r} in {self.path}"
                )
            parent = element.find("parent").attrib["link"]
            child = element.find("child").attrib["link"]
            axis_element = element.find("axis")
            axis = _parse_vector(
                None if axis_element is None else axis_element.attrib.get("xyz"),
                default=(1, 0, 0),
            )
            joint = _Joint(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                origin=_origin_transform(element.find("origin")),
                axis=axis,
            )
            if child in self._joint_by_child:
                raise ValueError(f"URDF link {child!r} has more than one parent joint")
            self._joint_by_child[child] = joint
            self.edges.append((parent, child))
            if joint_type != "fixed":
                self.movable_joint_names.append(name)

        child_links = set(self._joint_by_child)
        roots = links - child_links
        if len(roots) != 1:
            raise ValueError(f"Expected one URDF root link in {self.path}, found {sorted(roots)}")
        self.root_link = roots.pop()
        self.movable_joint_names = tuple(self.movable_joint_names)
        self.edges = tuple(self.edges)

    def transform(self, target_link, joint_positions):
        chain = []
        current_link = target_link
        while current_link != self.root_link:
            try:
                joint = self._joint_by_child[current_link]
            except KeyError as exc:
                raise ValueError(
                    f"Link {target_link!r} is not connected to root {self.root_link!r}"
                ) from exc
            chain.append(joint)
            current_link = joint.parent

        transform = np.eye(4, dtype=np.float64)
        for joint in reversed(chain):
            transform = transform @ joint.origin
            if joint.joint_type != "fixed":
                try:
                    angle = joint_positions[joint.name]
                except KeyError as exc:
                    raise ValueError(f"Missing position for joint {joint.name!r}") from exc
                transform = transform @ _axis_angle_transform(joint.axis, angle)
        return transform


def _joint_vector(values, allowed_shapes, *, label):
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape not in allowed_shapes:
        expected = " or ".join(str(shape) for shape in allowed_shapes)
        raise ValueError(f"{label} must have shape {expected}, got {vector.shape}")
    vector = vector.reshape(-1)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    return vector


def _rigid_transform(value, *, label):
    transform = np.asarray(value, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f"{label} must have homogeneous last row [0, 0, 0, 1]")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{label} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{label} rotation determinant must be +1")
    return transform.copy()


class WujiFingertipKinematics:
    """Estimate Wuji hand reference points in the RM75 base frame."""

    def __init__(self, arm_urdf=DEFAULT_ARM_URDF, hand_urdf=DEFAULT_HAND_URDF):
        self._arm = _UrdfTree(arm_urdf)
        self._hand = _UrdfTree(hand_urdf)
        if self._arm.root_link != "base_link":
            raise ValueError(f"RM75 URDF root must be 'base_link', got {self._arm.root_link!r}")
        if self._hand.root_link != "palm_link":
            raise ValueError(f"Wuji URDF root must be 'palm_link', got {self._hand.root_link!r}")
        if self._arm.movable_joint_names != ARM_JOINT_NAMES:
            raise ValueError(
                f"Unexpected RM75 joint order: {self._arm.movable_joint_names}"
            )
        if self._hand.movable_joint_names != HAND_JOINT_NAMES:
            raise ValueError(
                f"Unexpected Wuji joint order: {self._hand.movable_joint_names}"
            )

    def _prepare_state(
        self,
        arm_joint_angles,
        hand_joint_angles,
        flange_thickness_m,
        *,
        arm_angle_unit="degree",
        mount_to_palm_transform=None,
    ):
        arm_q = _joint_vector(
            arm_joint_angles, ((7,),), label="RM75 joint angles"
        )
        if arm_angle_unit == "degree":
            arm_q = np.deg2rad(arm_q)
        elif arm_angle_unit != "radian":
            raise ValueError("arm_angle_unit must be 'degree' or 'radian'")

        hand_q = _joint_vector(
            hand_joint_angles, ((20,), (5, 4)), label="Wuji joint angles"
        )
        thickness = float(flange_thickness_m)
        if not np.isfinite(thickness) or thickness < 0.0:
            raise ValueError("flange_thickness_m must be a finite non-negative value")

        if mount_to_palm_transform is None:
            mount_to_palm = np.eye(4, dtype=np.float64)
        else:
            mount_to_palm = _rigid_transform(
                mount_to_palm_transform, label="mount_to_palm_transform"
            )

        arm_positions = dict(zip(ARM_JOINT_NAMES, arm_q))
        hand_positions = dict(zip(HAND_JOINT_NAMES, hand_q))
        base_to_flange = self._arm.transform("link_7", arm_positions)

        flange_to_mount = np.eye(4, dtype=np.float64)
        flange_to_mount[2, 3] = thickness
        base_to_palm = base_to_flange @ flange_to_mount @ mount_to_palm
        return arm_positions, hand_positions, base_to_palm

    def arm_link_transform(
        self, arm_joint_angles, target_link="link_7", *, arm_angle_unit="degree"
    ):
        """Return one RM75 link transform in base_link using pure NumPy FK."""
        arm_q = _joint_vector(
            arm_joint_angles, ((7,),), label="RM75 joint angles"
        )
        if arm_angle_unit == "degree":
            arm_q = np.deg2rad(arm_q)
        elif arm_angle_unit != "radian":
            raise ValueError("arm_angle_unit must be 'degree' or 'radian'")
        arm_positions = dict(zip(ARM_JOINT_NAMES, arm_q))
        return self._arm.transform(target_link, arm_positions)

    def model_transforms(
        self,
        arm_joint_angles,
        hand_joint_angles,
        flange_thickness_m,
        *,
        arm_angle_unit="degree",
        mount_to_palm_transform=None,
    ):
        """Return all RM75 and Wuji link transforms expressed in ``base_link``."""
        arm_positions, hand_positions, base_to_palm = self._prepare_state(
            arm_joint_angles,
            hand_joint_angles,
            flange_thickness_m,
            arm_angle_unit=arm_angle_unit,
            mount_to_palm_transform=mount_to_palm_transform,
        )
        arm_transforms = {
            link_name: (
                np.eye(4, dtype=np.float64)
                if link_name == self._arm.root_link
                else self._arm.transform(link_name, arm_positions)
            )
            for link_name in self._arm.link_names
        }
        hand_transforms = {
            link_name: (
                base_to_palm
                if link_name == self._hand.root_link
                else base_to_palm @ self._hand.transform(link_name, hand_positions)
            )
            for link_name in self._hand.link_names
        }
        return {
            "arm": arm_transforms,
            "hand": hand_transforms,
            "arm_edges": self._arm.edges,
            "hand_edges": self._hand.edges,
        }

    def estimate(
        self,
        arm_joint_angles,
        hand_joint_angles,
        flange_thickness_m,
        *,
        arm_angle_unit="degree",
        mount_to_palm_transform=None,
        fingertip_offsets_m=None,
    ):
        """Return fingertip points in ``base_link`` as ``{finger: xyz_m}``.

        ``arm_joint_angles`` follows RealMan order ``joint_1..joint_7`` and is
        interpreted as degrees by default, matching ``rm_get_current_arm_state``.
        ``hand_joint_angles`` accepts either SDK shape ``(5, 4)`` or a flat
        20-value vector in thumb-to-little, proximal-to-distal order, in radians.

        The default mount model translates from ``link_7`` along its local +Z by
        ``flange_thickness_m`` and then aligns ``palm_link`` with that mounting
        frame. Supply ``mount_to_palm_transform`` for any additional measured
        translation or rotation. ``fingertip_offsets_m`` optionally specifies
        five local XYZ offsets, shape ``(5, 3)``, from each tip-link origin to the
        desired fingertip or finger-pad reference point.
        """
        transforms = self.model_transforms(
            arm_joint_angles,
            hand_joint_angles,
            flange_thickness_m,
            arm_angle_unit=arm_angle_unit,
            mount_to_palm_transform=mount_to_palm_transform,
        )

        if fingertip_offsets_m is None:
            fingertip_offsets = np.zeros((5, 3), dtype=np.float64)
        else:
            fingertip_offsets = np.asarray(fingertip_offsets_m, dtype=np.float64)
            if fingertip_offsets.shape != (5, 3) or not np.all(
                np.isfinite(fingertip_offsets)
            ):
                raise ValueError("fingertip_offsets_m must be a finite (5, 3) array")

        positions = {}
        for finger_name, link_name, local_offset in zip(
            FINGER_NAMES, FINGERTIP_LINK_NAMES, fingertip_offsets
        ):
            base_to_tip = transforms["hand"][link_name]
            positions[finger_name] = (
                base_to_tip[:3, 3] + base_to_tip[:3, :3] @ local_offset
            )
        return positions

    def estimate_keypoints(
        self,
        arm_joint_angles,
        hand_joint_angles,
        flange_thickness_m,
        *,
        keypoint_specs,
        arm_angle_unit="degree",
        mount_to_palm_transform=None,
    ):
        """Return configured Wuji keypoints in base_link as {name: xyz_m}.

        Each keypoint specification must provide name and link_name. Its optional
        offset_m is a local XYZ offset expressed in that link's coordinate frame.
        Extra visualization fields, such as color, are accepted and ignored.
        """
        transforms = self.model_transforms(
            arm_joint_angles,
            hand_joint_angles,
            flange_thickness_m,
            arm_angle_unit=arm_angle_unit,
            mount_to_palm_transform=mount_to_palm_transform,
        )
        hand_transforms = transforms["hand"]
        positions = {}

        for raw_spec in keypoint_specs:
            try:
                name = str(raw_spec["name"])
                link_name = str(raw_spec["link_name"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    "Each keypoint spec must provide 'name' and 'link_name'"
                ) from exc
            if not name or name in positions:
                raise ValueError(
                    f"Keypoint names must be non-empty and unique: {name!r}"
                )
            if link_name not in hand_transforms:
                raise ValueError(f"Unknown Wuji keypoint link: {link_name!r}")

            local_offset = np.asarray(
                raw_spec.get("offset_m", (0.0, 0.0, 0.0)), dtype=np.float64
            )
            if local_offset.shape != (3,) or not np.all(np.isfinite(local_offset)):
                raise ValueError(
                    f"Keypoint {name!r} offset_m must be a finite 3-vector"
                )

            base_to_link = hand_transforms[link_name]
            positions[name] = (
                base_to_link[:3, 3] + base_to_link[:3, :3] @ local_offset
            )

        return positions


@lru_cache(maxsize=1)
def _default_kinematics():
    return WujiFingertipKinematics()


def estimate_wuji_fingertip_positions(
    arm_joint_angles,
    hand_joint_angles,
    flange_thickness_m,
    *,
    arm_angle_unit="degree",
    mount_to_palm_transform=None,
    fingertip_offsets_m=None,
):
    """Convenience wrapper using the repository's RM75 and right-Wuji URDFs."""
    return _default_kinematics().estimate(
        arm_joint_angles,
        hand_joint_angles,
        flange_thickness_m,
        arm_angle_unit=arm_angle_unit,
        mount_to_palm_transform=mount_to_palm_transform,
        fingertip_offsets_m=fingertip_offsets_m,
    )


def estimate_wuji_keypoint_positions(
    arm_joint_angles,
    hand_joint_angles,
    flange_thickness_m,
    *,
    keypoint_specs,
    arm_angle_unit="degree",
    mount_to_palm_transform=None,
):
    """Estimate configured right-Wuji keypoints using the repository URDFs."""
    return _default_kinematics().estimate_keypoints(
        arm_joint_angles,
        hand_joint_angles,
        flange_thickness_m,
        keypoint_specs=keypoint_specs,
        arm_angle_unit=arm_angle_unit,
        mount_to_palm_transform=mount_to_palm_transform,
    )


__all__ = [
    "FINGER_NAMES",
    "FINGERTIP_LINK_NAMES",
    "WujiFingertipKinematics",
    "estimate_wuji_fingertip_positions",
    "estimate_wuji_keypoint_positions",
]
