"""Public API for the local FoundationPose C++ integration."""

from .artifacts import RunArtifacts
from .config import ObjectSpec, ProjectConfig
from .geometry import masked_cloud
from .runtime import FoundationPoseService, PipelineProcess
from .session import FoundationPoseSession
from .types import PointCloud, PointCloudConfig, PoseResult, RGBDFrame

__all__ = [
    "FoundationPoseSession",
    "FoundationPoseService",
    "masked_cloud",
    "ObjectSpec",
    "PipelineProcess",
    "PointCloud",
    "PointCloudConfig",
    "PoseResult",
    "ProjectConfig",
    "RGBDFrame",
    "RunArtifacts",
]

__version__ = "0.1.0"
