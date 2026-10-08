#!/usr/bin/env python3
"""SAM3 text- or point-prompt segmentation for one initialization image.

Runs in the server `sam3` conda env. For the offline pipeline, segment frame 0
to produce the initial mask used by FoundationPose register(); the subsequent
track_one() frames do NOT need masks.

Usage:
    python sam3_segment.py --input <rgb.png> --output <mask.png>
        [--ckpt sam3/weights/sam3.pt]
        [--prompt "red apple"]
        [--device cuda]
        [--thr 0.5]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def build(ckpt, device="cuda", bpe_path=None, enable_inst_interactivity=False):
    """Load SAM3 once; returns model + processor."""
    model = build_sam3_image_model(
        checkpoint_path=ckpt, device=device, bpe_path=bpe_path,
        enable_inst_interactivity=enable_inst_interactivity)
    model.eval()
    processor = Sam3Processor(model, device=device)
    return model, processor


def segment(processor, img, prompt, thr=0.5):
    """img: PIL Image (or (H,W,3) uint8 RGB ndarray).
    Returns mask (H,W) uint8 + top score.
    NOTE: pass a PIL Image to set_image — SAM3's set_image misreads
    ndarray.shape[-2:] as (H,W), which is wrong for (H,W,3) arrays."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    state = processor.set_image(img)
    state = processor.set_text_prompt(prompt, state)
    state = processor.set_confidence_threshold(thr, state)
    masks = state["masks"]  # torch.bool [N,1,H,W]
    scores = state["scores"]
    if masks is None or masks.shape[0] == 0:
        return np.zeros((img.size[1], img.size[0]), np.uint8), 0.0
    best = int(scores.argmax())
    mask = masks[best, 0].cpu().numpy()  # bool
    return mask.astype(np.uint8), float(scores[best])


class PointSelectionUI:
    """OpenCV point-prompt editor backed by SAM3's instance predictor."""

    toolbar_height = 58

    def __init__(self, model, state, rgb):
        import cv2

        self.cv2 = cv2
        self.model = model
        self.state = state
        self.rgb = rgb
        self.points = []
        self.labels = []
        self.mask = np.zeros(rgb.shape[:2], np.uint8)
        self.score = 0.0
        self.mode = 1
        self.confirmed = False
        self.cancelled = False
        self.error = None
        self.window = "SAM3 target selection"
        self.buttons = [
            ("+", 1, 10, 70), ("-", 0, 78, 138), ("Undo", "undo", 146, 246),
            ("Clear", "clear", 254, 354), ("Confirm", "confirm", 362, 492),
            ("Cancel", "cancel", 500, 610),
        ]

    def predict(self):
        if not self.points:
            self.mask.fill(0)
            self.score = 0.0
            return
        coords = np.asarray(self.points, dtype=np.float32)
        labels = np.asarray(self.labels, dtype=np.int32)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks, scores, _ = self.model.predict_inst(
                self.state, point_coords=coords, point_labels=labels,
                multimask_output=len(self.points) == 1)
        best = int(np.asarray(scores).argmax())
        self.mask = (np.asarray(masks[best]) > 0).astype(np.uint8)
        self.score = float(np.asarray(scores)[best])

    def apply_action(self, action):
        if action in (0, 1):
            self.mode = action
            return
        if action == "undo" and self.points:
            self.points.pop()
            self.labels.pop()
            self.predict()
        elif action == "clear":
            self.points.clear()
            self.labels.clear()
            self.predict()
        elif action == "confirm" and np.any(self.mask):
            self.confirmed = True
        elif action == "cancel":
            self.cancelled = True

    def on_mouse(self, event, x, y, _flags, _userdata):
        try:
            if event == self.cv2.EVENT_LBUTTONDOWN and y < self.toolbar_height:
                for _label, action, left, right in self.buttons:
                    if left <= x < right:
                        self.apply_action(action)
                        return
            if y < self.toolbar_height:
                return
            if event not in (self.cv2.EVENT_LBUTTONDOWN, self.cv2.EVENT_RBUTTONDOWN):
                return
            image_y = y - self.toolbar_height
            if not (0 <= x < self.rgb.shape[1] and 0 <= image_y < self.rgb.shape[0]):
                return
            label = 0 if event == self.cv2.EVENT_RBUTTONDOWN else self.mode
            self.points.append((x, image_y))
            self.labels.append(label)
            self.predict()
        except Exception as exc:  # surface callback failures in the main loop
            self.error = exc

    def render(self):
        cv2 = self.cv2
        view = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)
        if np.any(self.mask):
            tint = np.zeros_like(view)
            tint[..., 1] = 230
            selected = self.mask.astype(bool)
            view[selected] = cv2.addWeighted(view[selected], 0.5, tint[selected], 0.5, 0)
            contours, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(view, contours, -1, (255, 255, 255), 2)
        for (x, y), label in zip(self.points, self.labels):
            if label:
                cv2.circle(view, (x, y), 7, (255, 255, 255), 2)
                cv2.circle(view, (x, y), 4, (0, 220, 0), -1)
            else:
                cv2.line(view, (x - 7, y - 7), (x + 7, y + 7), (255, 255, 255), 5)
                cv2.line(view, (x - 7, y + 7), (x + 7, y - 7), (255, 255, 255), 5)
                cv2.line(view, (x - 6, y - 6), (x + 6, y + 6), (0, 0, 230), 3)
                cv2.line(view, (x - 6, y + 6), (x + 6, y - 6), (0, 0, 230), 3)
        toolbar = np.full((self.toolbar_height, view.shape[1], 3), 36, np.uint8)
        for label, action, left, right in self.buttons:
            active = action == self.mode
            color = (70, 120, 70) if active else (62, 62, 62)
            if action == "confirm" and not np.any(self.mask):
                color = (43, 43, 43)
            cv2.rectangle(toolbar, (left, 9), (right, 48), color, -1)
            cv2.rectangle(toolbar, (left, 9), (right, 48), (125, 125, 125), 1)
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0]
            cv2.putText(toolbar, label, (left + (right - left - size[0]) // 2, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1,
                        cv2.LINE_AA)
        status = f"+{sum(self.labels)}  -{len(self.labels) - sum(self.labels)}  score {self.score:.3f}"
        cv2.putText(toolbar, status, (625, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (220, 220, 220), 1, cv2.LINE_AA)
        return np.vstack((toolbar, view))

    def run(self):
        cv2 = self.cv2
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_mouse)
        print("Click mode: left click uses +/- toolbar mode; right click is negative. "
              "Enter confirms, U undoes, C clears, Esc cancels.", flush=True)
        try:
            while not self.confirmed and not self.cancelled:
                if self.error is not None:
                    raise RuntimeError("SAM3 point prediction failed") from self.error
                cv2.imshow(self.window, self.render())
                key = cv2.waitKey(20) & 0xFF
                if key in (13, 10) and np.any(self.mask):
                    self.confirmed = True
                elif key in (ord("u"), 8):
                    self.apply_action("undo")
                elif key == ord("c"):
                    self.apply_action("clear")
                elif key == 27:
                    self.cancelled = True
                if cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) < 1:
                    self.cancelled = True
        finally:
            cv2.destroyWindow(self.window)
        if not self.confirmed:
            raise RuntimeError("SAM3 point selection was cancelled")
        return self.mask, self.score


def save_result(rgb, mask, output):
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(out)
    overlay = rgb.copy()
    selected = mask.astype(bool)
    overlay[selected] = (0.5 * overlay[selected] + 0.5 * np.array([0, 230, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(str(out.with_name(out.stem + "_overlay.png")))
    print(f"mask -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    sam3_root = Path(__file__).resolve().parents[1] / "sam3"
    ap.add_argument("--ckpt", default=str(sam3_root / "weights/sam3.pt"))
    ap.add_argument("--bpe", default=str(sam3_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    ap.add_argument("--prompt", default="red apple")
    ap.add_argument("--prompts", default=None,
                    help="comma-separated fallback list, tried in one model load; "
                         "the highest-scored non-empty mask wins")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--mode", choices=("text", "click"), default="text")
    args = ap.parse_args()

    rgb = np.array(Image.open(args.input).convert("RGB"))
    print(f"image: {rgb.shape} {rgb.dtype}", flush=True)

    model, processor = build(args.ckpt, args.device, args.bpe,
                             enable_inst_interactivity=args.mode == "click")
    print("model loaded", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.mode == "click":
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(rgb))
        mask, score = PointSelectionUI(model, state, rgb).run()
        print(f"confirmed point mask: score {score:.4f}, mask sum {mask.sum()}", flush=True)
    else:
        prompts = ([p.strip() for p in args.prompts.split(",")]
                   if args.prompts else [args.prompt])
        # SAM3's forward pass is authored under bf16 autocast (see official example)
        best = None
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for p in prompts:
                mask, score = segment(processor, rgb, p, args.thr)
                print(f"  prompt '{p}': score {score:.4f}, mask sum {mask.sum()}", flush=True)
                nonempty = mask.sum() > 0
                best_empty = best is None or best[2].sum() == 0
                if best is None or (nonempty and (best_empty or score > best[1])):
                    best = (p, score, mask)
        prompt_used, score, mask = best
        print(f"best prompt: '{prompt_used}' score: {score:.4f}, mask sum: {mask.sum()}", flush=True)
    if not np.any(mask):
        raise RuntimeError("SAM3 produced an empty mask")
    save_result(rgb, mask, args.output)


if __name__ == "__main__":
    main()
