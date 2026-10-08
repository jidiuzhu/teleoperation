"""Synchronous in-process API for an already-running FoundationPose C++ service."""
import socket
import threading

import numpy as np

from .geometry import (ReferenceAccumulator, empty_cloud, load_obj_surface,
                       observed_cloud, transform_cloud)
from .protocol import send_frame, send_stop
from .types import PointCloudConfig, PoseResult


class FoundationPoseSession:
    """Register and track caller-provided RGB-D frames.

    Point-cloud generation is disabled when ``pointcloud_config`` is ``None``.
    This pose-only mode avoids all client-side projection and mesh work.
    """

    def __init__(self, host="127.0.0.1", port=50051, *, mesh_path=None,
                 pointcloud_config: PointCloudConfig | None = None,
                 pointcloud_every=1, connect_timeout=60.0,
                 register_timeout=300.0, track_timeout=60.0):
        if pointcloud_every < 1:
            raise ValueError("pointcloud_every must be at least 1")
        if pointcloud_config is not None and mesh_path is None:
            raise ValueError("mesh_path is required when point clouds are enabled")
        self.host = host
        self.port = int(port)
        self.connect_timeout = float(connect_timeout)
        self.register_timeout = float(register_timeout)
        self.track_timeout = float(track_timeout)
        self.pointcloud_config = pointcloud_config
        self.pointcloud_every = int(pointcloud_every)
        self._connection = None
        self._initialized = False
        self._next_frame_id = 0
        self._shape = None
        self._lock = threading.Lock()
        self._bounds = self._mesh_model = self._reference = None
        if pointcloud_config is not None:
            self._bounds, self._mesh_model = load_obj_surface(
                mesh_path, pointcloud_config.mesh_points)
            self._reference = ReferenceAccumulator(
                pointcloud_config.reference_frames,
                pointcloud_config.reference_voxel_m)

    @property
    def initialized(self):
        return self._initialized

    def connect(self):
        if self._connection is None:
            self._connection = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout)
            self._connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return self

    def register(self, frame, mask):
        with self._lock:
            if self._initialized:
                raise RuntimeError("session is already initialized")
            self.connect()
            self._connection.settimeout(self.register_timeout)
            frame_id = 0 if frame.frame_id is None else int(frame.frame_id)
            pose, inference_ms = send_frame(
                self._connection, frame_id, frame, mask=mask, initialize=True)
            self._initialized = True
            self._next_frame_id = frame_id + 1
            self._shape = np.shape(frame.depth_mm)
            return self._result(frame_id, frame, pose, inference_ms, mask, True, True)

    def track(self, frame, *, with_pointcloud=None):
        with self._lock:
            if not self._initialized:
                raise RuntimeError("register() must be called before track()")
            if np.shape(frame.depth_mm) != self._shape:
                raise ValueError("RGB-D frame dimensions changed after registration")
            frame_id = self._next_frame_id if frame.frame_id is None else int(frame.frame_id)
            self._connection.settimeout(self.track_timeout)
            pose, inference_ms = send_frame(self._connection, frame_id, frame)
            self._next_frame_id = frame_id + 1
            scheduled = frame_id % self.pointcloud_every == 0
            update_clouds = scheduled if with_pointcloud is None else bool(with_pointcloud)
            return self._result(
                frame_id, frame, pose, inference_ms, None, False, update_clouds)

    def _result(self, frame_id, frame, pose, inference_ms, mask, initialized, update_clouds):
        if self.pointcloud_config is None or not update_clouds:
            empty = empty_cloud()
            return PoseResult(frame_id, pose, inference_ms, empty, empty, empty,
                              initialized, False)
        observed = observed_cloud(
            frame, pose, self._bounds, self.pointcloud_config, mask=mask)
        self._reference.add(observed, pose)
        mesh = transform_cloud(self._mesh_model, pose, (255, 65, 50))
        reference = self._reference.at_pose(pose)
        return PoseResult(frame_id, pose, inference_ms, observed, mesh, reference,
                          initialized, True)

    def close(self):
        with self._lock:
            if self._connection is not None:
                try:
                    send_stop(self._connection, self._next_frame_id)
                except OSError:
                    pass
                self._connection.close()
                self._connection = None
            self._initialized = False

    def __enter__(self):
        return self.connect()

    def __exit__(self, _type, _value, _traceback):
        self.close()
