import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import torch

_HF_CACHE_DIR = tempfile.TemporaryDirectory(prefix="teleopwuji-hf-test-")
os.environ["HF_HOME"] = _HF_CACHE_DIR.name
os.environ["HF_DATASETS_CACHE"] = str(Path(_HF_CACHE_DIR.name) / "datasets")

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import convert_h5_to_act as converter


def _create_dataset_at_path(h5_file, path, value):
    parent, _, name = path.rpartition("/")
    group = h5_file.require_group(parent) if parent else h5_file
    group.create_dataset(name, data=value)


def create_synthetic_recording(path: Path, frames: int = 3) -> None:
    height, width, source_points = 8, 12, 5
    with h5py.File(path, "w") as dst:
        dst.attrs["recording_complete"] = True
        dst.attrs["avg_fps"] = 15.0
        dst.attrs["hand_dof"] = 20
        dst.attrs["hand_joint_names"] = json.dumps([f"finger_joint_{index}" for index in range(20)])
        dst.attrs["task_name"] = "synthetic grasp"
        T_base_camera = np.eye(4, dtype=np.float64)
        T_base_camera[:3, 3] = [1.0, 2.0, 3.0]
        dst.attrs["kinect_T_base_camera"] = T_base_camera

        rgb = np.zeros((frames, height, width, 3), dtype=np.uint8)
        rgb[..., 0] = 64
        depth = np.full((frames, height, width), 825, dtype=np.uint16)
        depth[:, 0, 0] = 0
        distances = np.tile(np.linspace(0.01, 0.15, 15), (frames, 1)).astype(np.float64)
        clouds = np.zeros((frames, source_points, 3), dtype=np.float32)
        clouds[:, :3] = [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5]]

        values = {
            "time": np.arange(frames, dtype=np.float64) / 15.0,
            "observations/qpos": np.tile(np.arange(27, dtype=np.float32), (frames, 1)),
            "observations/images/KINECT/rgb": rgb,
            "observations/images/KINECT/depth": depth,
            "observations/distances": distances,
            "observations/kinect/pointcloud_xyz": clouds,
            "observations/kinect/pointcloud_count": np.full(frames, 3, dtype=np.int32),
            "observations/kinect/valid": np.ones(frames, dtype=np.uint8),
            "observations/kinect/segmentation_valid": np.ones(frames, dtype=np.uint8),
            "action/arm_target_q": np.tile(np.arange(7, dtype=np.float32), (frames, 1)),
            "action/hand_command_q": np.tile(np.arange(20, dtype=np.float32), (frames, 1)),
        }
        for key, value in values.items():
            _create_dataset_at_path(dst, key, value)


class RecordingDataset:
    def __init__(self):
        self.frames = []
        self.saved_episodes = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1


class ConvertH5ToActTest(unittest.TestCase):
    def test_depth_is_clipped_encoded_and_repeated_to_three_channels(self):
        depth = np.array([[0, 150], [825, 1500]], dtype=np.uint16)
        image = converter.depth_to_resnet_image(depth, 2, 2, 0.15, 1.50)

        self.assertEqual(tuple(image.shape), (3, 2, 2))
        self.assertEqual(image.dtype, torch.uint8)
        self.assertTrue(torch.equal(image[0], image[1]))
        self.assertEqual(int(image[0, 0, 0]), 0)
        self.assertEqual(int(image[0, 0, 1]), 0)
        self.assertEqual(int(image[0, 1, 1]), 255)

    def test_point_cloud_uses_valid_prefix_repeats_and_moves_to_base(self):
        points = np.array(
            [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5], [99, 99, 99]],
            dtype=np.float32,
        )
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]

        result = converter.resample_and_transform_point_cloud(points, 3, 5, transform)

        self.assertEqual(result.shape, (5, 3))
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.5])
        np.testing.assert_allclose(result[3], result[0])
        self.assertFalse(np.any(result == 99))

    def test_episode_conversion_builds_all_model_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "episode.h5"
            create_synthetic_recording(source)
            info = converter.validate_episode(source, task_override="Pick up the apple")
            dataset = RecordingDataset()
            config = converter.ConversionConfig(image_height=8, image_width=12, point_count=8)

            converter.convert_episode(dataset, info, config)

        self.assertEqual(dataset.saved_episodes, 1)
        self.assertEqual(len(dataset.frames), 3)
        first = dataset.frames[0]
        self.assertEqual(tuple(first[converter.RGB_KEY].shape), (3, 8, 12))
        self.assertEqual(tuple(first[converter.DEPTH_IMAGE_KEY].shape), (3, 8, 12))
        self.assertEqual(first[converter.STATE_KEY].shape, (27,))
        self.assertEqual(first[converter.ENVIRONMENT_STATE_KEY].shape, (15,))
        self.assertEqual(first[converter.POINT_CLOUD_KEY].shape, (8, 3))
        self.assertEqual(first[converter.ACTION_KEY].shape, (27,))
        self.assertEqual(first["task"], "Pick up the apple")
        np.testing.assert_allclose(
            first[converter.ENVIRONMENT_STATE_KEY],
            first[converter.DISTANCE_KEY],
        )

    def test_invalid_segmentation_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "episode.h5"
            create_synthetic_recording(source)
            with h5py.File(source, "r+") as h5_file:
                h5_file["observations/kinect/segmentation_valid"][1] = 0

            with self.assertRaisesRegex(ValueError, "invalid segmentation"):
                converter.validate_episode(source)

    def test_empty_segmented_cloud_is_rejected_without_pose_dependency(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "episode.h5"
            create_synthetic_recording(source)
            with h5py.File(source, "r+") as h5_file:
                h5_file[converter.SOURCE_KEYS["point_count"]][1] = 0

            with self.assertRaisesRegex(ValueError, "pointcloud_count"):
                converter.validate_episode(source)

    def test_lerobot_dataset_is_created_and_reloadable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            source = temp_path / "episode.h5"
            target = temp_path / "lerobot_dataset"
            create_synthetic_recording(source)
            info = converter.validate_episode(source)
            config = converter.ConversionConfig(
                image_height=8,
                image_width=12,
                point_count=8,
                vcodec="h264",
            )

            output = converter.create_dataset(
                episodes=[info],
                repo_id="local/synthetic_wuji_act",
                target_dir=target,
                config=config,
                push_to_hub=False,
            )
            loaded = LeRobotDataset(repo_id="local/synthetic_wuji_act", root=output)

            self.assertEqual(loaded.num_episodes, 1)
            self.assertEqual(loaded.num_frames, 3)
            self.assertEqual(loaded.meta.fps, 15)
            self.assertEqual(tuple(loaded.features[converter.ACTION_KEY]["shape"]), (27,))
            self.assertEqual(
                tuple(loaded.features[converter.POINT_CLOUD_KEY]["shape"]),
                (8, 3),
            )
            self.assertTrue((output / "conversion_manifest.json").is_file())

    def test_actbh_forward_accepts_every_model_facing_modality(self):
        from lerobot.datasets.utils import dataset_to_policy_features
        from lerobot.policies.actbh.configuration_actbh import ACTBHConfig
        from lerobot.policies.actbh.modeling_actbh import ACTBHPolicy

        conversion_config = converter.ConversionConfig(
            image_height=32,
            image_width=32,
            point_count=8,
        )
        dataset_features = converter.build_features(
            conversion_config,
            tuple(f"finger_joint_{index}" for index in range(20)),
        )
        policy_features = dataset_to_policy_features(dataset_features)
        model_input_keys = {
            converter.STATE_KEY,
            converter.ENVIRONMENT_STATE_KEY,
            converter.POINT_CLOUD_KEY,
            converter.RGB_KEY,
            converter.DEPTH_IMAGE_KEY,
        }
        config = ACTBHConfig(
            chunk_size=2,
            n_action_steps=2,
            dim_model=64,
            n_heads=8,
            dim_feedforward=128,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_vae_encoder_layers=1,
            pretrained_backbone_weights=None,
            device="cpu",
        )
        config.input_features = {key: policy_features[key] for key in model_input_keys}
        config.output_features = {converter.ACTION_KEY: policy_features[converter.ACTION_KEY]}
        policy = ACTBHPolicy(config)
        policy.train()

        batch = {
            converter.STATE_KEY: torch.zeros(1, 27),
            converter.ENVIRONMENT_STATE_KEY: torch.zeros(1, 15),
            converter.POINT_CLOUD_KEY: torch.zeros(1, 8, 3),
            converter.RGB_KEY: torch.zeros(1, 3, 32, 32),
            converter.DEPTH_IMAGE_KEY: torch.zeros(1, 3, 32, 32),
            converter.ACTION_KEY: torch.zeros(1, 2, 27),
            "action_is_pad": torch.zeros(1, 2, dtype=torch.bool),
        }

        loss, loss_dict = policy(batch)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("l1_loss", loss_dict)


if __name__ == "__main__":
    unittest.main()
