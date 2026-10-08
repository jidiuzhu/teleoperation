"""Per-frame SAM2 refinement guided by an online Cutie mask."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np


def validate_tracking_mask(
    mask,
    expected_shape,
    *,
    min_pixels,
    reference_area=None,
    max_area_change=3.0,
):
    """Return a binary mask after rejecting empty or implausible predictions."""
    binary = np.asarray(np.asarray(mask) > 0, dtype=np.uint8)
    if binary.shape != tuple(expected_shape):
        raise ValueError(
            f"SAM2 mask shape {binary.shape} does not match frame {tuple(expected_shape)}"
        )
    area = int(binary.sum())
    if area < int(min_pixels):
        raise ValueError(f"SAM2 mask has only {area} pixels")
    if reference_area is not None:
        ratio = max(area, reference_area) / max(1, min(area, reference_area))
        if ratio > float(max_area_change):
            raise ValueError(f"SAM2 mask area differs from Cutie by {ratio:.2f}x")
    return binary


def expanded_mask_box(mask, *, padding_ratio=0.10):
    """Return a clipped XYXY box expanded around a non-empty Cutie mask."""
    binary = np.asarray(mask) > 0
    if binary.ndim != 2 or not np.any(binary):
        raise ValueError("Cannot derive a SAM2 box from an empty Cutie mask")
    if not 0 <= float(padding_ratio) <= 1:
        raise ValueError("SAM2 box padding ratio must be within [0, 1]")

    height, width = binary.shape
    ys, xs = np.nonzero(binary)
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    padding_x = int(round((x1 - x0 + 1) * float(padding_ratio)))
    padding_y = int(round((y1 - y0 + 1) * float(padding_ratio)))
    if padding_ratio > 0:
        padding_x = max(2, padding_x)
        padding_y = max(2, padding_y)
    return np.array(
        [
            max(0, x0 - padding_x),
            max(0, y0 - padding_y),
            min(width - 1, x1 + padding_x),
            min(height - 1, y1 + padding_y),
        ],
        dtype=np.float32,
    )


def select_mask_candidate(
    masks,
    scores,
    cutie_mask,
    *,
    min_pixels,
    max_area_change,
):
    """Choose the valid SAM2 candidate most consistent with the Cutie guide."""
    candidates = np.asarray(masks)
    confidence = np.asarray(scores, dtype=np.float32).reshape(-1)
    guide = np.asarray(cutie_mask) > 0
    if candidates.ndim == 2:
        candidates = candidates[None, ...]
    if candidates.ndim != 3 or candidates.shape[1:] != guide.shape:
        raise ValueError("SAM2 candidates do not match the Cutie mask shape")
    if len(confidence) != len(candidates):
        raise ValueError("SAM2 candidate scores do not match candidate masks")

    guide_area = int(guide.sum())
    guide_yx = np.argwhere(guide).mean(axis=0)
    diagonal = max(1.0, float(np.hypot(*guide.shape)))
    best = None
    rejection_messages = []
    for index, candidate in enumerate(candidates):
        try:
            binary = validate_tracking_mask(
                candidate,
                guide.shape,
                min_pixels=min_pixels,
                reference_area=guide_area,
                max_area_change=max_area_change,
            )
        except ValueError as exc:
            rejection_messages.append(str(exc))
            continue
        area = int(binary.sum())
        intersection = int(np.logical_and(binary, guide).sum())
        union = int(np.logical_or(binary, guide).sum())
        iou = intersection / max(1, union)
        center_shift = float(
            np.linalg.norm(np.argwhere(binary).mean(axis=0) - guide_yx) / diagonal
        )
        area_penalty = abs(float(np.log(area / max(1, guide_area))))
        model_score = float(confidence[index])
        if not np.isfinite(model_score):
            rejection_messages.append("SAM2 candidate has a non-finite score")
            continue
        rank = model_score + 2.0 * iou - 0.5 * center_shift - 0.15 * area_penalty
        if best is None or rank > best[0]:
            best = (rank, index, binary)
    if best is None:
        details = rejection_messages[0] if rejection_messages else "no candidates"
        raise ValueError(f"SAM2 produced no valid refinement ({details})")
    return best[2], best[1]


class Sam2BoxRefiner:
    """Refine a current-frame Cutie mask using only its expanded box prompt."""

    def __init__(
        self,
        checkpoint,
        *,
        model_config="configs/sam2.1/sam2.1_hiera_t.yaml",
        box_padding=0.10,
        crop_padding=0.50,
        min_mask_pixels=500,
        max_area_change=3.0,
        device="cuda",
        use_amp=True,
    ):
        if not 0 <= box_padding < crop_padding <= 1:
            raise ValueError(
                "SAM2 padding must satisfy 0 <= box_padding < crop_padding <= 1"
            )
        if min_mask_pixels < 1 or max_area_change <= 1:
            raise ValueError("Invalid SAM2 mask validation settings")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM2 Tiny checkpoint not found: {self.checkpoint}")
        self.model_config = str(model_config)
        self.box_padding = float(box_padding)
        self.crop_padding = float(crop_padding)
        self.min_mask_pixels = int(min_mask_pixels)
        self.max_area_change = float(max_area_change)
        self.device_name = str(device)
        self.use_amp = bool(use_amp)
        self.torch = None
        self.device = None
        self.model = None
        self.predictor = None
        self.frame_shape = None
        self.last_box = None
        self.last_crop_box = None
        self.last_crop_shape = None

    def load(self):
        if self.predictor is not None:
            return
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise RuntimeError(
                "SAM2 runtime is unavailable. Run "
                "foundationpose-kinect-0.1.0/tools/install_sam2_tiny.sh first"
            ) from exc
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SAM2 realtime refinement requires CUDA")
        device = torch.device(self.device_name)
        if device.type == "cuda" and torch.cuda.get_device_properties(device).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        model = build_sam2(
            self.model_config,
            str(self.checkpoint),
            device=device,
            apply_postprocessing=True,
        )
        self.torch = torch
        self.device = device
        self.model = model
        self.predictor = SAM2ImagePredictor(model)

    def _autocast(self):
        if self.device.type != "cuda":
            return nullcontext()
        return self.torch.autocast(
            device_type="cuda",
            dtype=self.torch.bfloat16,
            enabled=self.use_amp,
        )

    @staticmethod
    def _rgb(color_bgr):
        color = np.asarray(color_bgr)
        if color.dtype != np.uint8 or color.ndim != 3 or color.shape[2] != 3:
            raise ValueError("SAM2 color frame must be HxWx3 uint8 BGR")
        return np.ascontiguousarray(color[..., ::-1])

    def initialize(self, color_bgr, cutie_mask):
        self.load()
        self.frame_shape = np.asarray(color_bgr).shape[:2]
        return self.refine(color_bgr, cutie_mask)

    def refine(self, color_bgr, cutie_mask):
        if self.predictor is None or self.frame_shape is None:
            raise RuntimeError("SAM2 refiner must be initialized first")
        guide = validate_tracking_mask(
            cutie_mask,
            self.frame_shape,
            min_pixels=self.min_mask_pixels,
            max_area_change=self.max_area_change,
        )
        color = np.asarray(color_bgr)
        if color.shape[:2] != self.frame_shape:
            raise ValueError("Live frame dimensions differ from SAM2 initialization")
        self.last_box = expanded_mask_box(guide, padding_ratio=self.box_padding)
        self.last_crop_box = expanded_mask_box(
            guide,
            padding_ratio=self.crop_padding,
        )
        x0, y0, x1, y1 = self.last_crop_box.astype(np.int32)
        crop_rgb = self._rgb(color[y0 : y1 + 1, x0 : x1 + 1])
        crop_guide = np.ascontiguousarray(guide[y0 : y1 + 1, x0 : x1 + 1])
        if crop_rgb.size == 0 or crop_guide.size == 0:
            raise ValueError("SAM2 refinement crop is empty")
        crop_height, crop_width = crop_guide.shape
        self.last_crop_shape = (crop_height, crop_width)
        prompt_x0, prompt_y0, prompt_x1, prompt_y1 = self.last_box
        crop_box = np.array(
            [
                prompt_x0 - x0,
                prompt_y0 - y0,
                prompt_x1 - x0,
                prompt_y1 - y0,
            ],
            dtype=np.float32,
        )
        if np.allclose(crop_box, [0, 0, crop_width - 1, crop_height - 1]):
            raise ValueError("SAM2 prompt box has no background context inside its ROI")

        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with self.torch.inference_mode(), self._autocast():
            self.predictor.set_image(crop_rgb)
            masks, scores, _logits = self.predictor.predict(
                box=crop_box,
                multimask_output=False,
            )
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0
        crop_mask, _index = select_mask_candidate(
            masks,
            scores,
            crop_guide,
            min_pixels=self.min_mask_pixels,
            max_area_change=self.max_area_change,
        )
        mask = np.zeros(self.frame_shape, dtype=np.uint8)
        mask[y0 : y1 + 1, x0 : x1 + 1] = crop_mask
        return mask, inference_ms

    def close(self):
        predictor, model, torch = self.predictor, self.model, self.torch
        self.predictor = None
        self.model = None
        self.torch = None
        self.device = None
        self.last_box = None
        self.last_crop_box = None
        self.last_crop_shape = None
        del predictor, model
        if torch is not None and self.device_name == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


class CutieSam2Tracker:
    """Use Cutie for temporal tracking and SAM2 for current-frame refinement."""

    def __init__(self, cutie_tracker, sam2_refiner):
        self.cutie_tracker = cutie_tracker
        self.sam2_refiner = sam2_refiner
        self.last_box = None
        self.last_cutie_mask = None
        self.cutie_inference_ms = 0.0
        self.sam2_inference_ms = 0.0

    def initialize(self, color_bgr, initial_mask):
        try:
            cutie_mask = self.cutie_tracker.initialize(color_bgr, initial_mask)
            refined, sam2_ms = self.sam2_refiner.initialize(color_bgr, cutie_mask)
        except BaseException:
            self.close()
            raise
        self.last_cutie_mask = cutie_mask
        self.last_box = self.sam2_refiner.last_box
        self.sam2_inference_ms = sam2_ms
        return refined

    def track(self, color_bgr):
        self.last_box = None
        cutie_mask, cutie_ms = self.cutie_tracker.track(color_bgr)
        self.last_cutie_mask = cutie_mask
        refined, sam2_ms = self.sam2_refiner.refine(color_bgr, cutie_mask)
        self.last_box = self.sam2_refiner.last_box
        self.cutie_inference_ms = cutie_ms
        self.sam2_inference_ms = sam2_ms
        return refined, cutie_ms + sam2_ms

    def close(self):
        try:
            self.sam2_refiner.close()
        finally:
            self.cutie_tracker.close()
