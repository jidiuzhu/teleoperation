#!/usr/bin/env python3
"""Replay synchronized Wuji HDF5 images, segmented cloud and signals in Rerun."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import rerun as rr


def replay(path: Path, frame_step: int = 1) -> None:
    if frame_step < 1:
        raise ValueError("frame_step must be positive")
    with h5py.File(path, "r") as source:
        count = len(source["observations/qpos"])
        names = json.loads(source.attrs.get("distance_keypoint_names", "[]"))
        camera_transform = source.attrs.get("kinect_T_base_camera")
        if camera_transform is not None:
            camera_transform = np.asarray(camera_transform)
        if camera_transform is not None and (
            camera_transform.shape != (4, 4) or not np.isfinite(camera_transform).all()
        ):
            raise ValueError("kinect_T_base_camera must be 4x4")
        rr.init("RM75_Wuji_HDF5_Replay", spawn=True)
        for index in range(0, count, frame_step):
            rr.set_time("frame", sequence=index)
            if "time" in source:
                rr.set_time("recording", duration=float(source["time"][index]))
            for joint_index, angle in enumerate(source["observations/qpos"][index]):
                rr.log(f"signals/joints/{joint_index:02d}", rr.Scalars(float(angle)))
            if "observations/distances" in source:
                distances = source["observations/distances"][index]
                for distance_index, distance in enumerate(distances):
                    if np.isfinite(distance):
                        name = names[distance_index] if distance_index < len(names) else str(distance_index)
                        rr.log(f"signals/distances/{name}", rr.Scalars(float(distance)))
            images = source.get("observations/images/KINECT")
            if images is not None:
                if "rgb" in images:
                    rr.log("world/kinect/rgb", rr.Image(images["rgb"][index]))
                if "depth" in images:
                    rr.log("world/kinect/depth", rr.DepthImage(images["depth"][index], meter=1000))
            kinect = source.get("observations/kinect")
            if kinect is not None and "pointcloud_xyz" in kinect:
                points = kinect["pointcloud_xyz"][index]
                valid_count = int(kinect["pointcloud_count"][index])
                points = points[:valid_count]
                if len(points) and np.isfinite(points).all():
                    if camera_transform is not None:
                        points = points @ camera_transform[:3, :3].T + camera_transform[:3, 3]
                    cloud_path = "world/base/segmented_object" if camera_transform is not None else "world/kinect/segmented_object"
                    rr.log(cloud_path, rr.Points3D(points))
                else:
                    rr.log("world/base/segmented_object", rr.Clear(recursive=False))
                    rr.log("world/kinect/segmented_object", rr.Clear(recursive=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_file", type=Path)
    parser.add_argument("--frame-step", type=int, default=1)
    args = parser.parse_args()
    replay(args.h5_file, args.frame_step)


if __name__ == "__main__":
    main()
