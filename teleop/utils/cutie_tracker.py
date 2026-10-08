"""Online single-object mask tracking with the external Cutie VOS model."""

from __future__ import annotations

from contextlib import nullcontext
import inspect
from pathlib import Path
import time

import numpy as np


CUTIE_WEIGHTS_MD5 = "a6071de6136982e396851903ab4c083a"


def validate_tracking_mask(mask, expected_shape, *, min_pixels, previous_area=None,
                           max_area_change=3.0):
    binary = np.asarray(np.asarray(mask) > 0, dtype=np.uint8)
    if binary.shape != tuple(expected_shape):
        raise ValueError(
            f"Cutie mask shape {binary.shape} does not match frame {tuple(expected_shape)}"
        )
    area = int(binary.sum())
    if area < int(min_pixels):
        raise ValueError(f"Cutie mask has only {area} pixels")
    if previous_area is not None:
        ratio = max(area, previous_area) / max(1, min(area, previous_area))
        if ratio > float(max_area_change):
            raise ValueError(f"Cutie mask area changed {ratio:.2f}x")
    return binary


class CutieTracker:
    """Maintain Cutie temporal memory and return a mask for each incoming frame."""

    def __init__(
        self,
        weights,
        *,
        max_internal_size=480,
        mem_every=10,
        min_mask_pixels=500,
        max_area_change=3.0,
        device="cuda",
        use_amp=True,
    ):
        if max_internal_size == 0 or max_internal_size < -1:
            raise ValueError("Cutie max_internal_size must be -1 or positive")
        if mem_every < 1 or min_mask_pixels < 1 or max_area_change <= 1:
            raise ValueError("Invalid Cutie memory or mask validation settings")

        self.weights = Path(weights).expanduser().resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"Cutie weights not found: {self.weights}")
        self.max_internal_size = int(max_internal_size)
        self.mem_every = int(mem_every)
        self.min_mask_pixels = int(min_mask_pixels)
        self.max_area_change = float(max_area_change)
        self.device_name = str(device)
        self.use_amp = bool(use_amp)
        self.network = None
        self.processor = None
        self.torch = None
        self.frame_shape = None
        self.previous_area = None
        self.frame_count = 0

    def load(self):
        if self.processor is not None:
            return
        try:
            import torch
            from hydra import compose, initialize_config_dir
            from omegaconf import open_dict
            from cutie.inference.inference_core import InferenceCore
            from cutie.model.cutie import CUTIE
            from cutie.model.utils import resnet
        except ImportError as exc:
            raise RuntimeError(
                "Cutie runtime is unavailable. Install the pinned Cutie package and "
                "minimal dependencies documented in teleop/README_wuji.md"
            ) from exc

        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Cutie requires CUDA for the realtime publisher")
        device = torch.device(self.device_name)
        config_dir = Path(inspect.getfile(CUTIE)).parents[1] / "config"
        with initialize_config_dir(
            version_base="1.3.2", config_dir=str(config_dir)
        ):
            cfg = compose(config_name="video_config")
        with open_dict(cfg):
            cfg.mem_every = self.mem_every
            cfg.max_internal_size = self.max_internal_size
            cfg.amp = self.use_amp

        # The full Cutie checkpoint replaces both encoders. Avoid redundant
        # ImageNet downloads performed by the upstream constructors.
        original_resnet18 = resnet.resnet18
        original_resnet50 = resnet.resnet50
        resnet.resnet18 = lambda pretrained=True, extra_dim=0, model_dir=None: (
            original_resnet18(False, extra_dim, model_dir)
        )
        resnet.resnet50 = lambda pretrained=True, extra_dim=0, model_dir=None: (
            original_resnet50(False, extra_dim, model_dir)
        )
        try:
            network = CUTIE(cfg).to(device).eval()
        finally:
            resnet.resnet18 = original_resnet18
            resnet.resnet50 = original_resnet50

        state = torch.load(self.weights, map_location=device, weights_only=True)
        network.load_weights(state)
        processor = InferenceCore(network, cfg=cfg)
        processor.max_internal_size = self.max_internal_size
        self.torch = torch
        self.network = network
        self.processor = processor

    def _image_tensor(self, color_bgr):
        color = np.asarray(color_bgr)
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("Cutie color frame must be HxWx3 uint8 BGR")
        rgb = np.ascontiguousarray(color[..., ::-1])
        return (
            self.torch.from_numpy(rgb)
            .permute(2, 0, 1)
            .to(self.network.device, dtype=self.torch.float32)
            .div_(255.0)
        )

    def _autocast(self):
        if self.network.device.type != "cuda":
            return nullcontext()
        return self.torch.amp.autocast("cuda", enabled=self.use_amp)

    def initialize(self, color_bgr, mask):
        self.load()
        self.frame_shape = np.asarray(color_bgr).shape[:2]
        binary = validate_tracking_mask(
            mask,
            self.frame_shape,
            min_pixels=self.min_mask_pixels,
            max_area_change=self.max_area_change,
        )
        image = self._image_tensor(color_bgr)
        mask_tensor = self.torch.from_numpy(binary.astype(np.int64)).to(
            self.network.device
        )
        with self.torch.inference_mode(), self._autocast():
            output = self.processor.step(image, mask_tensor, objects=[1])
        tracked = self.processor.output_prob_to_mask(output).cpu().numpy()
        tracked = validate_tracking_mask(
            tracked,
            self.frame_shape,
            min_pixels=self.min_mask_pixels,
            max_area_change=self.max_area_change,
        )
        self.previous_area = int(tracked.sum())
        self.frame_count = 1
        return tracked

    def track(self, color_bgr):
        if self.processor is None or self.frame_shape is None:
            raise RuntimeError("Cutie tracker must be initialized with the clicked mask")
        if np.asarray(color_bgr).shape[:2] != self.frame_shape:
            raise ValueError("Live frame dimensions differ from Cutie initialization")
        started = time.perf_counter()
        image = self._image_tensor(color_bgr)
        with self.torch.inference_mode(), self._autocast():
            output = self.processor.step(image)
        mask = self.processor.output_prob_to_mask(output).cpu().numpy()
        inference_ms = (time.perf_counter() - started) * 1000.0
        mask = validate_tracking_mask(
            mask,
            self.frame_shape,
            min_pixels=self.min_mask_pixels,
            previous_area=self.previous_area,
            max_area_change=self.max_area_change,
        )
        self.previous_area = int(mask.sum())
        self.frame_count += 1
        return mask, inference_ms

    def close(self):
        processor, network, torch = self.processor, self.network, self.torch
        self.processor = None
        self.network = None
        self.torch = None
        del processor, network
        if torch is not None and self.device_name == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
