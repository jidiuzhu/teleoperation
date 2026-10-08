#!/usr/bin/env python3
"""Segment a model-free RGB sequence with one SAM3 model lifetime."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from .sam3_segment import PointSelectionUI, build, save_result, segment
except ImportError:  # Direct execution adds server/ rather than the project root.
    from sam3_segment import PointSelectionUI, build, save_result, segment


def best_text_mask(processor, rgb, prompts, threshold):
    best = None
    for prompt in prompts:
        mask, score = segment(processor, rgb, prompt, threshold)
        if best is None or (np.any(mask) and (not np.any(best[2]) or score > best[1])):
            best = (prompt, score, mask)
    return best


def propagated_points(previous):
    ys, xs = np.nonzero(previous)
    center = (float(np.median(xs)), float(np.median(ys)))
    height, width = previous.shape
    pad = max(12, int(0.1 * max(xs.max() - xs.min(), ys.max() - ys.min())))
    negatives = [
        (max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)),
        (min(width - 1, int(xs.max()) + pad), max(0, int(ys.min()) - pad)),
        (max(0, int(xs.min()) - pad), min(height - 1, int(ys.max()) + pad)),
        (min(width - 1, int(xs.max()) + pad), min(height - 1, int(ys.max()) + pad)),
    ]
    coords = np.asarray([center, *negatives], np.float32)
    labels = np.asarray([1, 0, 0, 0, 0], np.int32)
    return coords, labels


def automatic_point_mask(model, processor, rgb, previous):
    coords, labels = propagated_points(previous)
    state = processor.set_image(Image.fromarray(rgb))
    candidates = []
    for candidate_coords, candidate_labels in ((coords[:1], labels[:1]), (coords, labels)):
        masks, scores, _ = model.predict_inst(
            state, point_coords=candidate_coords, point_labels=candidate_labels,
            multimask_output=True)
        masks = masks.detach().cpu().numpy() if torch.is_tensor(masks) else np.asarray(masks)
        scores = scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        for mask, score in zip(masks, scores):
            candidate = (np.squeeze(mask) > 0).astype(np.uint8)
            candidates.append((temporal_mask_cost(previous, candidate, float(score)),
                               candidate, float(score), candidate_coords, candidate_labels))
    _, mask, score, selected_coords, selected_labels = min(candidates, key=lambda item: item[0])
    return mask, score, state, selected_coords, selected_labels


def temporal_mask_cost(previous, candidate, score):
    previous_area = int(previous.sum())
    candidate_area = int(candidate.sum())
    if candidate_area == 0:
        return np.inf
    area_cost = abs(np.log(candidate_area / previous_area))
    intersection = int(np.logical_and(previous, candidate).sum())
    union = int(np.logical_or(previous, candidate).sum())
    overlap_cost = 1.0 - intersection / max(1, union)
    previous_y, previous_x = np.nonzero(previous)
    candidate_y, candidate_x = np.nonzero(candidate)
    diagonal = max(1.0, np.hypot(previous_x.max() - previous_x.min(),
                                 previous_y.max() - previous_y.min()))
    center_shift = np.hypot(np.median(candidate_x) - np.median(previous_x),
                            np.median(candidate_y) - np.median(previous_y)) / diagonal
    return area_cost + 1.5 * overlap_cost + 0.5 * center_shift - 0.05 * score


def correct_point_mask(model, state, rgb, mask, score, coords, labels, reason):
    print(f"Point propagation needs correction: {reason}", flush=True)
    ui = PointSelectionUI(model, state, rgb)
    ui.points = [(int(round(point[0])), int(round(point[1]))) for point in coords]
    ui.labels = labels.astype(int).tolist()
    ui.mask = mask
    ui.score = score
    return ui.run()


def write_manifest(sequence, mode, records):
    output = sequence / "segmentation.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"mode": mode, "frames": records}, indent=2),
                         encoding="utf-8")
    temporary.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--bpe", required=True)
    parser.add_argument("--mode", choices=("text", "click"), default="text")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--area-change-ratio", type=float, default=2.5)
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--resume", action="store_true",
                        help="continue after the last consecutive saved mask")
    args = parser.parse_args()

    sequence = Path(args.sequence)
    rgb_paths = sorted((sequence / "rgb").glob("*.png"))
    if len(rgb_paths) < 3:
        raise RuntimeError(f"At least 3 RGB frames are required in {sequence / 'rgb'}")
    mask_dir = sequence / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    prompts = [value.strip() for value in args.prompts.split(",") if value.strip()]
    if not prompts:
        raise RuntimeError("At least one non-empty SAM3 prompt is required")

    records = []
    previous = None
    start_index = 0
    if args.resume:
        for index, rgb_path in enumerate(rgb_paths):
            existing = mask_dir / rgb_path.name
            if not existing.is_file():
                start_index = index
                break
            mask = (np.asarray(Image.open(existing).convert("L")) > 0).astype(np.uint8)
            area = int(mask.sum())
            if area < args.min_pixels:
                start_index = index
                break
            records.append({"frame": rgb_path.stem, "score": None, "pixels": area,
                            "method": "existing-mask"})
            previous = mask
        else:
            write_manifest(sequence, args.mode, records)
            print(f"All {len(records)} masks already exist; SAM3 is not loaded.")
            return

    model, processor = build(args.ckpt, "cuda", args.bpe,
                             enable_inst_interactivity=args.mode == "click")
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for index in range(start_index, len(rgb_paths)):
            rgb_path = rgb_paths[index]
            rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
            if args.mode == "text":
                prompt, score, mask = best_text_mask(
                    processor, rgb, prompts, args.threshold)
            elif previous is None:
                state = processor.set_image(Image.fromarray(rgb))
                mask, score = PointSelectionUI(model, state, rgb).run()
                prompt = "interactive-points"
            else:
                mask, score, state, coords, labels = automatic_point_mask(
                    model, processor, rgb, previous)
                prompt = "propagated-positive-negative-points"

            area = int(mask.sum())
            problem = None
            if area < args.min_pixels:
                problem = f"mask has only {area} pixels at {rgb_path.name}"
            if previous is not None:
                previous_area = int(previous.sum())
                change = max(area, previous_area) / max(1, min(area, previous_area))
                if change > args.area_change_ratio:
                    problem = (f"mask area changed {change:.2f}x at {rgb_path.name} "
                               f"({previous_area} -> {area} pixels)")
            if problem and args.mode == "click" and previous is not None:
                mask, score = correct_point_mask(
                    model, state, rgb, mask, score, coords, labels, problem)
                area = int(mask.sum())
                if area < args.min_pixels:
                    raise RuntimeError(
                        f"Confirmed SAM3 mask is too small at {rgb_path.name}: {area}")
                prompt = "interactive-correction"
            elif problem:
                raise RuntimeError(problem + "; correct the capture or use click mode")
            output = mask_dir / rgb_path.name
            save_result(rgb, mask, output)
            records.append({"frame": rgb_path.stem, "score": score, "pixels": area,
                            "method": prompt})
            write_manifest(sequence, args.mode, records)
            previous = mask
            print(f"SEGMENT {index + 1}/{len(rgb_paths)} score={score:.4f} pixels={area}",
                  flush=True)

    print(f"Segmented {len(records)} frames. SAM3 is now exiting.")


if __name__ == "__main__":
    main()
