"""Small, dependency-free reader for the shared project JSON."""
import json
from dataclasses import dataclass
from pathlib import Path

from .types import PointCloudConfig


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    mesh_path: Path
    prompts: tuple[str, ...]
    source: str = "cad"


class ProjectConfig:
    def __init__(self, path, project_root=None):
        self.path = Path(path).resolve()
        self.project_root = (Path(project_root).resolve() if project_root else
                             self.path.parent.parent.resolve())
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def selected_object(self):
        return self.data["selected_object"]

    def object(self, name=None, *, require_mesh=True):
        name = name or self.selected_object
        try:
            value = self.data["objects"][name]
        except KeyError as exc:
            raise KeyError(f"object is not configured: {name}") from exc
        mesh = Path(value["mesh"])
        if not mesh.is_absolute():
            mesh = self.project_root / mesh
        mesh = mesh.resolve()
        if require_mesh and not mesh.is_file():
            raise FileNotFoundError(f"object mesh does not exist: {mesh}")
        prompts = tuple(str(item).strip() for item in value["prompts"] if str(item).strip())
        if not prompts:
            raise ValueError(f"object has no prompts: {name}")
        return ObjectSpec(name, mesh, prompts, value.get("source", "cad"))

    def pointcloud_config(self):
        value = self.data.get("pointcloud", {})
        return PointCloudConfig(
            stride=int(value.get("stride", 3)),
            min_depth_m=float(value.get("min_depth_m", 0.15)),
            max_depth_m=float(value.get("max_depth_m", 2.0)),
            max_points=int(value.get("max_points", 120000)),
            mesh_points=int(value.get("mesh_points", 12000)),
            reference_frames=int(value.get("reference_frames", 5)),
            reference_voxel_m=float(value.get("reference_voxel_m", 0.003)),
            crop_margin_m=float(value.get("crop_margin_m", 0.005)),
            mask_erosion_px=int(value.get("mask_erosion_px", 0)),
        )

    @property
    def port(self):
        return int(self.data.get("runtime", {}).get("port", 50051))
