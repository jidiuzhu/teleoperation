"""Read atomic results produced by the managed launcher."""
import json
from pathlib import Path

import numpy as np

from .types import PointCloud


_PLY_VERTEX = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("red", "u1"), ("green", "u1"), ("blue", "u1"),
], align=False)


def load_binary_xyzrgb_ply(path):
    path = Path(path)
    with path.open("rb") as stream:
        count = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"incomplete PLY header: {path}")
            text = line.decode("ascii").strip()
            if text == "format binary_little_endian 1.0":
                pass
            elif text.startswith("element vertex "):
                count = int(text.split()[-1])
            elif text == "end_header":
                break
        if count is None:
            raise ValueError(f"PLY vertex count is missing: {path}")
        vertices = np.fromfile(stream, dtype=_PLY_VERTEX, count=count)
    if len(vertices) != count:
        raise ValueError(f"truncated PLY payload: {path}")
    points = np.column_stack((vertices["x"], vertices["y"], vertices["z"]))
    colors = np.column_stack((vertices["red"], vertices["green"], vertices["blue"]))
    return PointCloud(points.astype(np.float32, copy=False),
                      colors.astype(np.uint8, copy=False), "camera")


class RunArtifacts:
    """Polling-friendly view over one ``test_output/live/run_*`` directory."""

    def __init__(self, run_directory):
        self.path = Path(run_directory).resolve()

    @property
    def pose_path(self):
        return self.path / "latest_pose.txt"

    def latest_pose(self):
        pose = np.loadtxt(self.pose_path, dtype=np.float32)
        if pose.shape != (4, 4):
            raise ValueError(f"invalid latest pose: {self.pose_path}")
        return pose

    def latest_cloud(self, kind="observed"):
        if kind not in ("observed", "mesh", "reference"):
            raise ValueError("kind must be observed, mesh, or reference")
        paths = sorted((self.path / "pointcloud" / kind).glob("*.ply"))
        if not paths:
            return None
        return load_binary_xyzrgb_ply(paths[-1])

    def metrics(self):
        path = self.path / "metrics.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    @classmethod
    def latest(cls, output_root):
        paths = sorted(Path(output_root).glob("run_*"))
        if not paths:
            return None
        return cls(paths[-1])

