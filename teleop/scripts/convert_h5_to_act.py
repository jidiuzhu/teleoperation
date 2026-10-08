#!/usr/bin/env python3
"""Convert Wuji teleoperation HDF5 recordings to a multimodal LeRobot ACT dataset.

The generated dataset is designed for this repository's ``actbh`` policy, an
ACT/CVAE variant with a PointNet token. It also remains usable by vanilla ACT
when the point-cloud field is omitted from the policy inputs.

Model-facing features:

* ``observation.images.KINECT_rgb``: resized RGB video.
* ``observation.images.KINECT_depth``: metric depth clipped to a configurable
  range, quantized to uint8, and repeated to three channels for ResNet.
* ``observation.state``: 7 measured RM75 joints + 20 measured Wuji joints.
* ``observation.environment_state``: 15 base-frame hand-to-object distances.
* ``observation.point_cloud_global``: object cloud transformed from the Kinect
  camera frame to ``base_link`` and deterministically resized to 1024 points.
* ``action``: 7 RM75 target joints + 20 filtered/limited Wuji command joints.

Example:

    conda run -n lerobot python teleop/scripts/convert_h5_to_act.py \
      --source teleop/data_records_wuji \
      --include actdata \
      --repo-id local/apple10_wuji_act \
      --target-dir teleop/lerobot_datasets/apple10_wuji_act \
      --task "Grasp the apple"
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING

import h5py
import numpy as np
from tqdm import tqdm

TELEOP_ROOT = Path(__file__).resolve().parents[1]
if str(TELEOP_ROOT) not in sys.path:
    sys.path.insert(0, str(TELEOP_ROOT))

from utils.actbh_observation import (
    ACTION_KEY,
    DEPTH_IMAGE_KEY,
    DISTANCE_KEY,
    ENVIRONMENT_STATE_KEY,
    POINT_CLOUD_KEY,
    RGB_KEY,
    STATE_KEY,
    depth_to_resnet_image,
    require_rigid_transform as _require_rigid_transform,
    resample_and_transform_point_cloud,
    resize_rgb,
)

if TYPE_CHECKING:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset


SOURCE_KEYS = {
    "time": "time",
    "qpos": "observations/qpos",
    "rgb": "observations/images/KINECT/rgb",
    "depth": "observations/images/KINECT/depth",
    "distances": "observations/distances",
    "point_cloud": "observations/kinect/pointcloud_xyz",
    "point_count": "observations/kinect/pointcloud_count",
    "kinect_valid": "observations/kinect/valid",
    "arm_action": "action/arm_target_q",
    "hand_action": "action/hand_command_q",
}


@dataclass(frozen=True)
class ConversionConfig:
    fps: int = 15
    image_height: int = 240
    image_width: int = 320
    point_count: int = 1024
    depth_min_m: float = 0.15
    depth_max_m: float = 1.50
    vcodec: str = "h264"

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image dimensions must be positive")
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")
        if not 0 <= self.depth_min_m < self.depth_max_m:
            raise ValueError("depth range must satisfy 0 <= min < max")


@dataclass(frozen=True)
class EpisodeInfo:
    path: Path
    frames: int
    source_fps: float
    task: str
    hand_joint_names: tuple[str, ...]


def collect_source_files(
    source: Path,
    includes: list[str] | None = None,
    excludes: list[str] | None = None,
) -> list[Path]:
    if source.is_file():
        candidates = [source]
    elif source.is_dir():
        candidates = sorted(
            path
            for path in source.iterdir()
            if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}
        )
    else:
        raise FileNotFoundError(f"Source does not exist: {source}")

    include_terms = [term for term in includes or [] if term]
    exclude_terms = [term for term in excludes or [] if term]
    selected = []
    for path in candidates:
        if include_terms and not all(term in path.name for term in include_terms):
            continue
        if any(term in path.name for term in exclude_terms):
            continue
        selected.append(path)

    if not selected:
        raise FileNotFoundError(
            f"No HDF5 files selected from {source} with "
            f"include={include_terms or None}, exclude={exclude_terms or None}"
        )
    return selected


def _decode_json_string(value, *, label: str) -> tuple[str, ...]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    try:
        result = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in HDF5 attribute {label}: {value!r}") from exc
    if not isinstance(result, list) or not all(isinstance(item, str) for item in result):
        raise ValueError(f"HDF5 attribute {label} must be a JSON string list")
    return tuple(result)


def validate_episode(path: Path, task_override: str | None = None) -> EpisodeInfo:
    with h5py.File(path, "r") as src:
        missing = [key for key in SOURCE_KEYS.values() if key not in src]
        if missing:
            raise KeyError(f"{path.name} is missing required fields: {missing}")
        if "kinect_T_base_camera" not in src.attrs:
            raise KeyError(f"{path.name} is missing root attribute kinect_T_base_camera")
        _require_rigid_transform(src.attrs["kinect_T_base_camera"], label="kinect_T_base_camera")

        if not bool(src.attrs.get("recording_complete", False)):
            raise ValueError(f"{path.name} is not a complete recording")
        if int(src.attrs.get("hand_dof", -1)) != 20:
            raise ValueError(f"{path.name} must contain a 20-DoF Wuji hand")
        hand_joint_names = _decode_json_string(
            src.attrs.get("hand_joint_names", "[]"), label="hand_joint_names"
        )
        if len(hand_joint_names) != 20:
            raise ValueError(f"{path.name} has {len(hand_joint_names)} hand joint names, expected 20")

        frames = int(src[SOURCE_KEYS["time"]].shape[0])
        if frames <= 0:
            raise ValueError(f"{path.name} contains no frames")
        source_fps = float(src.attrs.get("avg_fps", 0.0))
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"{path.name} has an invalid or missing avg_fps attribute")
        expected_shapes = {
            "qpos": (frames, 27),
            "distances": (frames, 15),
            "point_count": (frames,),
            "kinect_valid": (frames,),
            "arm_action": (frames, 7),
            "hand_action": (frames, 20),
        }
        for logical_name, expected in expected_shapes.items():
            actual = src[SOURCE_KEYS[logical_name]].shape
            if actual != expected:
                raise ValueError(
                    f"{path.name}:{SOURCE_KEYS[logical_name]} has shape {actual}, expected {expected}"
                )

        rgb_shape = src[SOURCE_KEYS["rgb"]].shape
        depth_shape = src[SOURCE_KEYS["depth"]].shape
        point_cloud_shape = src[SOURCE_KEYS["point_cloud"]].shape
        if len(rgb_shape) != 4 or rgb_shape[0] != frames or rgb_shape[-1] != 3:
            raise ValueError(f"{path.name}: RGB data must have shape TxHxWx3, got {rgb_shape}")
        if depth_shape != rgb_shape[:3]:
            raise ValueError(
                f"{path.name}: depth shape {depth_shape} must match RGB spatial shape {rgb_shape[:3]}"
            )
        if (
            len(point_cloud_shape) != 3
            or point_cloud_shape[0] != frames
            or point_cloud_shape[-1] != 3
        ):
            raise ValueError(
                f"{path.name}: point cloud data must have shape TxNx3, got {point_cloud_shape}"
            )

        if not np.asarray(src[SOURCE_KEYS["kinect_valid"]][...], dtype=bool).all():
            raise ValueError(f"{path.name} contains invalid Kinect frames")
        if "observations/kinect/segmentation_valid" in src and not np.asarray(
            src["observations/kinect/segmentation_valid"][...], dtype=bool
        ).all():
            raise ValueError(f"{path.name} contains invalid segmentation frames")
        if not np.isfinite(src[SOURCE_KEYS["distances"]][...]).all():
            raise ValueError(f"{path.name} contains non-finite hand-object distances")
        for finite_key in ("qpos", "arm_action", "hand_action"):
            if not np.isfinite(src[SOURCE_KEYS[finite_key]][...]).all():
                raise ValueError(f"{path.name}:{SOURCE_KEYS[finite_key]} contains non-finite values")

        point_counts = np.asarray(src[SOURCE_KEYS["point_count"]][...], dtype=np.int64)
        max_source_points = point_cloud_shape[1]
        if np.any(point_counts <= 0) or np.any(point_counts > max_source_points):
            raise ValueError(
                f"{path.name} has pointcloud_count outside [1, {max_source_points}]"
            )
        point_cloud = src[SOURCE_KEYS["point_cloud"]]
        for frame_index, source_count in enumerate(point_counts):
            if not np.isfinite(point_cloud[frame_index, :source_count]).all():
                raise ValueError(f"{path.name} point cloud frame {frame_index} is non-finite")

        source_task = src.attrs.get("task_name", path.stem)
        if isinstance(source_task, bytes):
            source_task = source_task.decode("utf-8")
        task = task_override.strip() if task_override and task_override.strip() else str(source_task)
        if not task:
            raise ValueError(f"{path.name} has an empty task description")

    return EpisodeInfo(
        path=path,
        frames=frames,
        source_fps=source_fps,
        task=task,
        hand_joint_names=hand_joint_names,
    )


def build_features(config: ConversionConfig, hand_joint_names: tuple[str, ...]) -> dict:
    arm_observation_names = [f"rm75_joint_{index + 1}" for index in range(7)]
    state_names = arm_observation_names + list(hand_joint_names)
    action_names = [f"rm75_joint_{index + 1}_target" for index in range(7)] + [
        f"{name}_command" for name in hand_joint_names
    ]
    distance_names = [f"hand_object_distance_{index}" for index in range(15)]

    return {
        ACTION_KEY: {"dtype": "float32", "shape": (27,), "names": action_names},
        STATE_KEY: {"dtype": "float32", "shape": (27,), "names": state_names},
        ENVIRONMENT_STATE_KEY: {
            "dtype": "float32",
            "shape": (15,),
            "names": distance_names,
        },
        DISTANCE_KEY: {"dtype": "float32", "shape": (15,), "names": distance_names},
        POINT_CLOUD_KEY: {
            "dtype": "float32",
            "shape": (config.point_count, 3),
            "names": ["point", "xyz"],
        },
        RGB_KEY: {
            "dtype": "video",
            "shape": (3, config.image_height, config.image_width),
            "names": ["channels", "height", "width"],
        },
        DEPTH_IMAGE_KEY: {
            "dtype": "video",
            "shape": (3, config.image_height, config.image_width),
            "names": ["channels", "height", "width"],
        },
    }


def convert_episode(
    dataset: "LeRobotDataset",
    episode: EpisodeInfo,
    config: ConversionConfig,
) -> None:
    with h5py.File(episode.path, "r") as src:
        T_base_camera = _require_rigid_transform(
            src.attrs["kinect_T_base_camera"], label="kinect_T_base_camera"
        )
        qpos = src[SOURCE_KEYS["qpos"]][...].astype(np.float32)
        arm_action = src[SOURCE_KEYS["arm_action"]][...].astype(np.float32)
        hand_action = src[SOURCE_KEYS["hand_action"]][...].astype(np.float32)
        distances = src[SOURCE_KEYS["distances"]][...].astype(np.float32)

        actions = np.concatenate([arm_action, hand_action], axis=1)
        iterator = tqdm(range(episode.frames), desc=episode.path.name, leave=False)
        for frame_index in iterator:
            source_point_count = int(src[SOURCE_KEYS["point_count"]][frame_index])
            point_cloud_base = resample_and_transform_point_cloud(
                src[SOURCE_KEYS["point_cloud"]][frame_index],
                source_count=source_point_count,
                target_count=config.point_count,
                T_base_camera=T_base_camera,
            )

            frame = {
                ACTION_KEY: actions[frame_index],
                STATE_KEY: qpos[frame_index],
                ENVIRONMENT_STATE_KEY: distances[frame_index],
                DISTANCE_KEY: distances[frame_index],
                POINT_CLOUD_KEY: point_cloud_base,
                RGB_KEY: resize_rgb(
                    src[SOURCE_KEYS["rgb"]][frame_index],
                    config.image_height,
                    config.image_width,
                ),
                DEPTH_IMAGE_KEY: depth_to_resnet_image(
                    src[SOURCE_KEYS["depth"]][frame_index],
                    config.image_height,
                    config.image_width,
                    config.depth_min_m,
                    config.depth_max_m,
                ),
                "task": episode.task,
            }
            dataset.add_frame(frame)

    dataset.save_episode()


def create_dataset(
    episodes: list[EpisodeInfo],
    repo_id: str,
    target_dir: Path | None,
    config: ConversionConfig,
    push_to_hub: bool,
) -> Path:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if target_dir is not None and target_dir.exists():
        raise FileExistsError(
            f"Target directory already exists: {target_dir}. Choose a new path to avoid overwriting data."
        )

    features = build_features(config, episodes[0].hand_joint_names)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=config.fps,
        features=features,
        root=str(target_dir) if target_dir is not None else None,
        vcodec=config.vcodec,
    )
    for episode in episodes:
        convert_episode(dataset, episode, config)
    dataset.finalize()

    output_root = Path(dataset.root)
    manifest = {
        "format": "wuji-segmented-cloud-actbh-v2",
        "repo_id": repo_id,
        "conversion": asdict(config),
        "coordinate_frames": {
            POINT_CLOUD_KEY: "base_link",
            DISTANCE_KEY: "base_link distances in meters",
        },
        "depth_encoding": {
            "source_units": "millimeter",
            "model_input": "linear metric depth clipped to [depth_min_m, depth_max_m], uint8, 3-channel",
            "invalid_value": 0,
        },
        "episodes": [
            {
                "source": str(info.path),
                "frames": info.frames,
                "source_fps": info.source_fps,
                "task": info.task,
            }
            for info in episodes
        ],
    }
    (output_root / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    if push_to_hub:
        dataset.push_to_hub()
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Wuji HDF5 episodes into a multimodal LeRobot ACT/ACTBH dataset."
    )
    parser.add_argument("--source", required=True, help="HDF5 file or directory containing HDF5 episodes")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset ID, for example local/apple10_wuji_act")
    parser.add_argument("--target-dir", default=None, help="Local output directory; it must not already exist")
    parser.add_argument("--task", default=None, help="Override the task text for all selected episodes")
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help="Require a filename substring; repeat to require multiple substrings",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Exclude a filename substring; repeat to exclude multiple substrings",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--image-height", type=int, default=240)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--point-count", type=int, default=1024)
    parser.add_argument("--depth-min-m", type=float, default=0.15)
    parser.add_argument("--depth-max-m", type=float, default=1.50)
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate selected source episodes without creating a LeRobot dataset",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ConversionConfig(
        fps=args.fps,
        image_height=args.image_height,
        image_width=args.image_width,
        point_count=args.point_count,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
        vcodec=args.vcodec,
    )
    config.validate()

    source = Path(args.source).expanduser().resolve()
    paths = collect_source_files(source, includes=args.include, excludes=args.exclude)
    episodes = [validate_episode(path, task_override=args.task) for path in paths]
    reference_joint_names = episodes[0].hand_joint_names
    for episode in episodes[1:]:
        if episode.hand_joint_names != reference_joint_names:
            raise ValueError(f"Hand joint order differs in {episode.path.name}")
    fps_tolerance = max(0.5, config.fps * 0.05)
    fps_mismatches = [
        episode
        for episode in episodes
        if abs(episode.source_fps - config.fps) > fps_tolerance
    ]
    if fps_mismatches:
        details = ", ".join(
            f"{episode.path.name}={episode.source_fps:.3f} Hz" for episode in fps_mismatches
        )
        raise ValueError(
            f"Requested --fps={config.fps} differs from source recording rates: {details}"
        )

    total_frames = sum(episode.frames for episode in episodes)
    print(f"Validated {len(episodes)} episode(s), {total_frames} frames at {config.fps} Hz")
    for episode in episodes:
        print(
            f"  {episode.path.name}: {episode.frames} frames, "
            f"{episode.source_fps:.3f} Hz, task={episode.task!r}"
        )
    if args.validate_only:
        return

    target_dir = Path(args.target_dir).expanduser().resolve() if args.target_dir else None
    output_root = create_dataset(
        episodes=episodes,
        repo_id=args.repo_id,
        target_dir=target_dir,
        config=config,
        push_to_hub=args.push_to_hub,
    )
    print(f"Converted dataset: {output_root}")
    print("Train all modalities with policy.type=actbh; vanilla act does not encode point clouds.")


if __name__ == "__main__":
    main()
