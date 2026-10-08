#!/usr/bin/env python3
"""Validate the local pipeline JSON and emit shell-safe resolved values."""
import argparse
import json
import shlex
from pathlib import Path


def fail(message):
    raise ValueError(message)


def require_number(value, name, minimum=None, maximum=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        fail(f"{name} must be a number")
    if integer and int(value) != value:
        fail(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        fail(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        fail(f"{name} must be <= {maximum}")
    return int(value) if integer else float(value)


def require_bool(value, name):
    if not isinstance(value, bool):
        fail(f"{name} must be true or false")
    return value


def resolve(config_path, root, object_override=None, allow_missing_mesh=False,
            everything=False):
    config_path = Path(config_path).resolve()
    root = Path(root).resolve()
    try:
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read config {config_path}: {exc}")

    if everything:
        if object_override is not None:
            fail("--everything cannot be combined with --object")
        object_name = "everything"
        mesh = None
        prompts = []
    else:
        objects = cfg.get("objects")
        if not isinstance(objects, dict) or not objects:
            fail("objects must be a non-empty object")
        object_name = object_override or cfg.get("selected_object")
        if object_name not in objects:
            fail(f"object '{object_name}' is unavailable; choose one of: {', '.join(objects)}")
        obj = objects[object_name]
        if not isinstance(obj, dict):
            fail(f"objects.{object_name} must be an object")
        mesh_value = obj.get("mesh")
        if not isinstance(mesh_value, str) or not mesh_value:
            fail(f"objects.{object_name}.mesh must be a path")
        mesh = Path(mesh_value)
        if not mesh.is_absolute():
            mesh = root / mesh
        mesh = mesh.resolve()
        if not allow_missing_mesh and not mesh.is_file():
            fail(f"mesh does not exist: {mesh}")
        prompts = obj.get("prompts")
        if not isinstance(prompts, list) or not prompts or not all(
                isinstance(value, str) and value.strip() for value in prompts):
            fail(f"objects.{object_name}.prompts must be a non-empty string list")

    segmentation = cfg.get("segmentation", {})
    mode = segmentation.get("mode", "text")
    if mode not in ("text", "click"):
        fail("segmentation.mode must be 'text' or 'click'")
    threshold = require_number(segmentation.get("threshold", 0.5),
                               "segmentation.threshold", 0.0, 1.0)
    checkpoint = Path(segmentation.get("checkpoint", "sam3/weights/sam3.pt"))
    bpe = Path(segmentation.get("bpe", "sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"))
    checkpoint = checkpoint if checkpoint.is_absolute() else root / checkpoint
    bpe = bpe if bpe.is_absolute() else root / bpe
    if not checkpoint.is_file():
        fail(f"SAM3 checkpoint does not exist: {checkpoint}")
    if not bpe.is_file():
        fail(f"SAM3 BPE file does not exist: {bpe}")

    camera = cfg.get("camera", {})
    hybrid = cfg.get("hybrid", {})
    model_free = cfg.get("model_free", {})
    runtime = cfg.get("runtime", {})
    visual = cfg.get("visualization", {})
    cloud = cfg.get("pointcloud", {})
    layout = visual.get("layout", "rgb")
    if layout not in ("rgb", "rgbd", "rgbd_cloud"):
        fail("visualization.layout must be rgb, rgbd, or rgbd_cloud")
    cloud_mode = cloud.get("mode", "all")
    if cloud_mode not in ("all", "observed", "mesh", "reference"):
        fail("pointcloud.mode must be all, observed, mesh, or reference")
    model_free_root = Path(model_free.get("output_root", "model_free_objects"))
    if not model_free_root.is_absolute():
        model_free_root = root / model_free_root
    result = {
        "object_name": object_name,
        "mesh": "" if mesh is None else str(mesh),
        "prompt": ",".join(value.strip() for value in prompts),
        "segmentation_mode": mode,
        "sam_threshold": threshold,
        "sam_checkpoint": str(checkpoint.resolve()),
        "sam_bpe": str(bpe.resolve()),
        "hybrid_enabled": require_bool(hybrid.get("enabled", False),
                                       "hybrid.enabled"),
        "hybrid_sam_hz": require_number(hybrid.get("sam_hz", 5.0),
                                        "hybrid.sam_hz", 0.1, 30.0),
        "hybrid_window_seconds": require_number(hybrid.get("window_seconds", 2.0),
                                                "hybrid.window_seconds", 0.2, 60.0),
        "hybrid_reregister_center_px": require_number(
            hybrid.get("reregister_center_error_px", 40.0),
            "hybrid.reregister_center_error_px", 1.0, 1000.0),
        "hybrid_reregister_cooldown": require_number(
            hybrid.get("reregister_cooldown_seconds", 2.0),
            "hybrid.reregister_cooldown_seconds", 0.0, 300.0),
        "hybrid_min_mask_pixels": require_number(
            hybrid.get("min_mask_pixels", 500), "hybrid.min_mask_pixels", 1,
            integer=True),
        "hybrid_max_area_change": require_number(
            hybrid.get("max_mask_area_change_ratio", 2.5),
            "hybrid.max_mask_area_change_ratio", 1.01, 100.0),
        "warmup": require_number(camera.get("warmup_frames", 10),
                                 "camera.warmup_frames", 0, integer=True),
        "model_free_root": str(model_free_root.resolve()),
        "model_free_capture_frames": require_number(
            model_free.get("capture_frames", 240), "model_free.capture_frames", 3,
            integer=True),
        "model_free_capture_every": require_number(
            model_free.get("capture_every_frames", 3),
            "model_free.capture_every_frames", 1, integer=True),
        "model_free_area_change": require_number(
            model_free.get("mask_area_change_ratio", 2.5),
            "model_free.mask_area_change_ratio", 1.01),
        "model_free_surface_depth_span": require_number(
            model_free.get("max_surface_depth_span_m", 0.2),
            "model_free.max_surface_depth_span_m", 0.03, 1.0),
        "model_free_voxel": require_number(model_free.get("voxel_m", 0.003),
                                             "model_free.voxel_m", 0.0005, 0.02),
        "model_free_icp_fitness": require_number(
            model_free.get("icp_fitness_min", 0.3), "model_free.icp_fitness_min",
            0.0, 1.0),
        "model_free_icp_rmse": require_number(
            model_free.get("icp_rmse_max_m", 0.015), "model_free.icp_rmse_max_m",
            0.0001, 0.1),
        "model_free_poisson_depth": require_number(
            model_free.get("poisson_depth", 8), "model_free.poisson_depth", 5, 11,
            integer=True),
        "model_free_max_triangles": require_number(
            model_free.get("max_triangles", 100000), "model_free.max_triangles", 1000,
            integer=True),
        "port": require_number(runtime.get("port", 50051), "runtime.port", 1, 65535,
                               integer=True),
        "max_frames": require_number(runtime.get("max_frames", 0),
                                     "runtime.max_frames", 0, integer=True),
        "headless": require_bool(runtime.get("headless", False), "runtime.headless"),
        "image": str(runtime.get("docker_image", "foundationpose-cpp:trt10-ubuntu22.04")),
        "visualization_layout": layout,
        "preview_scale": require_number(visual.get("preview_scale", 0.5),
                                        "visualization.preview_scale", 0.1, 1.0),
        "visualization_save_every": require_number(
            visual.get("save_every_frames", 30), "visualization.save_every_frames", 0,
            integer=True),
        "pointcloud_enabled": require_bool(cloud.get("enabled", True), "pointcloud.enabled"),
        "pointcloud_mode": cloud_mode,
        "pointcloud_save_every": require_number(cloud.get("save_every_frames", 30),
                                                "pointcloud.save_every_frames", 0, integer=True),
        "pointcloud_preview_every": require_number(cloud.get("preview_every_frames", 2),
                                                   "pointcloud.preview_every_frames", 1,
                                                   integer=True),
        "pointcloud_stride": require_number(cloud.get("stride", 3),
                                            "pointcloud.stride", 1, integer=True),
        "pointcloud_min_depth": require_number(cloud.get("min_depth_m", 0.15),
                                               "pointcloud.min_depth_m", 0.0),
        "pointcloud_max_depth": require_number(cloud.get("max_depth_m", 2.0),
                                               "pointcloud.max_depth_m", 0.001),
        "pointcloud_max_points": require_number(cloud.get("max_points", 120000),
                                                "pointcloud.max_points", 1, integer=True),
        "pointcloud_mesh_points": require_number(cloud.get("mesh_points", 12000),
                                                 "pointcloud.mesh_points", 100, integer=True),
        "pointcloud_reference_frames": require_number(cloud.get("reference_frames", 5),
                                                      "pointcloud.reference_frames", 1,
                                                      integer=True),
        "pointcloud_reference_voxel": require_number(cloud.get("reference_voxel_m", 0.003),
                                                     "pointcloud.reference_voxel_m", 0.0001),
        "pointcloud_crop_margin": require_number(cloud.get("crop_margin_m", 0.005),
                                                "pointcloud.crop_margin_m", 0.0),
        "pointcloud_yaw": require_number(cloud.get("view_yaw_deg", -25.0),
                                        "pointcloud.view_yaw_deg"),
        "pointcloud_pitch": require_number(cloud.get("view_pitch_deg", -20.0),
                                          "pointcloud.view_pitch_deg"),
    }
    if result["pointcloud_max_depth"] <= result["pointcloud_min_depth"]:
        fail("pointcloud.max_depth_m must be greater than min_depth_m")
    if not result["image"]:
        fail("runtime.docker_image must not be empty")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--object")
    parser.add_argument(
        "--everything", action="store_true",
        help="resolve model-independent click segmentation settings without an object mesh")
    parser.add_argument(
        "--allow-missing-mesh", action="store_true",
        help="resolve shared capture/reconstruction settings without requiring the selected mesh")
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    try:
        values = resolve(
            args.config,
            args.root,
            args.object,
            args.allow_missing_mesh,
            everything=args.everything,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.shell:
        for key, value in values.items():
            if isinstance(value, bool):
                value = int(value)
            print(f"cfg_{key}={shlex.quote(str(value))}")
    else:
        print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
