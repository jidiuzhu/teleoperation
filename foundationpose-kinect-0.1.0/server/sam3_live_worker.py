#!/usr/bin/env python3
"""Bounded, latest-frame SAM3 worker used by the live hybrid pipeline."""

import multiprocessing as mp
import queue
import time
import traceback
from collections import deque


def _put_latest(target, value):
    """Publish without allowing a slow consumer to grow an unbounded queue."""
    while True:
        try:
            target.put_nowait(value)
            return
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass


def _best_text_mask(processor, rgb, prompts, threshold):
    import numpy as np

    from server.sam3_segment import segment

    if not prompts:
        raise ValueError("at least one SAM3 text prompt is required")
    best = None
    for prompt in prompts:
        mask, score = segment(processor, rgb, prompt, threshold)
        nonempty = bool(np.any(mask))
        best_empty = best is None or not np.any(best[2])
        if best is None or (nonempty and (best_empty or score > best[1])):
            best = (prompt, score, mask)
    return best


def _mask_problem(mask, previous, min_pixels, max_area_change):
    import numpy as np

    area = int(np.count_nonzero(mask))
    if area < min_pixels:
        return f"mask has only {area} pixels"
    if previous is not None:
        previous_area = int(np.count_nonzero(previous))
        ratio = max(area, previous_area) / max(1, min(area, previous_area))
        if ratio > max_area_change:
            return f"mask area changed {ratio:.2f}x"
    return None


def _worker_main(requests, results, stop_event, options, seed_mask):
    try:
        import numpy as np
        import torch

        from server.sam3_batch_segment import automatic_point_mask
        from server.sam3_segment import build

        started = time.perf_counter()
        model, processor = build(
            options["checkpoint"], "cuda", options["bpe"],
            enable_inst_interactivity=True)
        model.eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.synchronize()
        load_ms = (time.perf_counter() - started) * 1000.0
        _put_latest(results, {
            "kind": "ready",
            "load_ms": load_ms,
            "gpu_allocated_mib": torch.cuda.memory_allocated() // (1024 * 1024),
            "gpu_reserved_mib": torch.cuda.memory_reserved() // (1024 * 1024),
        })

        history_size = max(1, int(round(
            options["sam_hz"] * options["window_seconds"])))
        mask_history = deque(maxlen=history_size)
        if seed_mask is not None and np.any(seed_mask):
            mask_history.append((time.monotonic(),
                                 np.asarray(seed_mask > 0, dtype=np.uint8)))

        while not stop_event.is_set():
            try:
                request = requests.get(timeout=0.1)
            except queue.Empty:
                continue
            if request is None:
                break

            frame_id, captured_at, rgb = request
            inference_started = time.perf_counter()
            try:
                cutoff = captured_at - options["window_seconds"]
                while mask_history and mask_history[0][0] < cutoff:
                    mask_history.popleft()
                previous = mask_history[-1][1] if mask_history else None
                method = "point-propagation"
                with torch.inference_mode(), torch.autocast(
                        "cuda", dtype=torch.bfloat16):
                    if previous is None:
                        if not options["prompts"]:
                            raise RuntimeError(
                                "segmentation seed mask expired before it could be propagated")
                        prompt, score, mask = _best_text_mask(
                            processor, rgb, options["prompts"], options["threshold"])
                        method = f"text:{prompt}"
                    else:
                        mask, score, _state, _coords, _labels = automatic_point_mask(
                            model, processor, rgb, previous)
                        problem = _mask_problem(
                            mask, previous, options["min_pixels"],
                            options["max_area_change"])
                        if problem and options["prompts"]:
                            prompt, score, mask = _best_text_mask(
                                processor, rgb, options["prompts"],
                                options["threshold"])
                            method = f"text-fallback:{prompt}"

                torch.cuda.synchronize()
                inference_ms = (time.perf_counter() - inference_started) * 1000.0
                problem = _mask_problem(
                    mask, previous, options["min_pixels"],
                    options["max_area_change"])
                if problem:
                    raise RuntimeError(problem)
                mask = np.asarray(mask > 0, dtype=np.uint8)
                mask_history.append((captured_at, mask))
                _put_latest(results, {
                    "kind": "mask",
                    "frame_id": frame_id,
                    "captured_at": captured_at,
                    "finished_at": time.monotonic(),
                    "mask": mask,
                    "score": float(score),
                    "pixels": int(mask.sum()),
                    "method": method,
                    "inference_ms": inference_ms,
                    "history_size": len(mask_history),
                    "history_capacity": history_size,
                })
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                _put_latest(results, {
                    "kind": "fatal",
                    "message": f"CUDA out of memory: {exc}",
                    "traceback": traceback.format_exc(),
                })
                return
            except Exception as exc:
                _put_latest(results, {
                    "kind": "frame_error",
                    "frame_id": frame_id,
                    "message": str(exc),
                })
    except Exception as exc:
        _put_latest(results, {
            "kind": "fatal",
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })


class LiveSamWorker:
    """Spawn SAM3 separately and keep only the newest queued camera frame."""

    def __init__(self, checkpoint, bpe, prompts, threshold, sam_hz,
                 window_seconds, min_pixels=500, max_area_change=2.5,
                 seed_mask=None):
        context = mp.get_context("spawn")
        self._requests = context.Queue(maxsize=1)
        self._results = context.Queue(maxsize=2)
        self._stop_event = context.Event()
        options = {
            "checkpoint": str(checkpoint),
            "bpe": str(bpe),
            "prompts": list(prompts),
            "threshold": float(threshold),
            "sam_hz": float(sam_hz),
            "window_seconds": float(window_seconds),
            "min_pixels": int(min_pixels),
            "max_area_change": float(max_area_change),
        }
        self._process = context.Process(
            target=_worker_main,
            args=(self._requests, self._results, self._stop_event,
                  options, seed_mask),
            name="sam3-live-worker",
            daemon=True)
        self._process.start()

    @property
    def alive(self):
        return self._process.is_alive()

    def submit(self, frame_id, captured_at, rgb):
        """Submit the newest frame; return True when an older queued frame was dropped."""
        dropped = False
        try:
            self._requests.put_nowait((frame_id, captured_at, rgb))
            return dropped
        except queue.Full:
            try:
                self._requests.get_nowait()
                dropped = True
            except queue.Empty:
                pass
        try:
            self._requests.put_nowait((frame_id, captured_at, rgb))
        except queue.Full:
            dropped = True
        return dropped

    def poll(self):
        messages = []
        while True:
            try:
                messages.append(self._results.get_nowait())
            except queue.Empty:
                return messages

    def close(self, timeout=5.0):
        self._stop_event.set()
        try:
            self._requests.put_nowait(None)
        except queue.Full:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                pass
            try:
                self._requests.put_nowait(None)
            except queue.Full:
                pass
        self._process.join(timeout=timeout)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=2.0)
        for channel in (self._requests, self._results):
            channel.cancel_join_thread()
            channel.close()
