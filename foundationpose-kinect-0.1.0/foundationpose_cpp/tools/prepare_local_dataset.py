#!/usr/bin/env python3
"""Convert this workspace's RGB/NPY-depth recording into the C++ demo layout."""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    recording = Path(args.recording).resolve()
    output = Path(args.output).resolve()
    for name in ("rgb", "depth", "masks"):
        (output / name).mkdir(parents=True, exist_ok=True)

    metadata = json.loads((recording / "meta.json").read_text(encoding="utf-8"))
    K = np.asarray(metadata["K"], dtype=np.float32)
    np.savetxt(output / "cam_K.txt", K, fmt="%.9g")
    shutil.copy2(args.mesh, output / "mesh.obj")

    rgb_paths = sorted((recording / "rgb").glob("*.png"))
    if args.limit > 0:
        rgb_paths = rgb_paths[: args.limit]
    for rgb_path in rgb_paths:
        frame_id = rgb_path.stem
        shutil.copy2(rgb_path, output / "rgb" / rgb_path.name)
        depth_m = np.load(recording / "depth" / f"{frame_id}.npy")
        depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
        if not cv2.imwrite(str(output / "depth" / f"{frame_id}.png"), depth_mm):
            raise RuntimeError(f"Failed to write depth frame {frame_id}")

    print(f"Prepared {len(rgb_paths)} frames in {output}")


if __name__ == "__main__":
    main()
