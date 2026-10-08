#!/usr/bin/env python3
"""Reconstruct a meter-scale mesh from masked RGB-D views of a rotating object."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


def load_cloud(rgb_path, depth_path, mask_path, K, stride, min_depth, max_depth,
               max_surface_span):
    bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = np.load(depth_path).astype(np.float32)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if bgr is None or mask is None or depth.shape != mask.shape or depth.shape != bgr.shape[:2]:
        raise RuntimeError(f"RGB/depth/mask geometry differs for {rgb_path.stem}")
    sampled_depth = depth[::stride, ::stride]
    sampled_mask = mask[::stride, ::stride] > 0
    rows, cols = np.indices(sampled_depth.shape, dtype=np.float32)
    u, v = cols * stride, rows * stride
    valid = (sampled_mask & np.isfinite(sampled_depth) &
             (sampled_depth >= min_depth) & (sampled_depth <= max_depth))
    foreground_depth = sampled_depth[valid]
    if len(foreground_depth) < 100:
        raise RuntimeError(f"Too few valid target depths for {rgb_path.stem}")
    median_depth = float(np.median(foreground_depth))
    mad = float(np.median(np.abs(foreground_depth - median_depth))) * 1.4826
    half_span = min(max_surface_span * 0.5, max(0.03, mad * 6.0))
    valid &= np.abs(sampled_depth - median_depth) <= half_span
    z = sampled_depth[valid]
    points = np.column_stack(((u[valid] - K[0, 2]) * z / K[0, 0],
                              (v[valid] - K[1, 2]) * z / K[1, 1], z))
    colors = bgr[::stride, ::stride, ::-1][valid].astype(np.float64) / 255.0
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    return cloud


def prepare(cloud, voxel):
    result = cloud.voxel_down_sample(voxel)
    if len(result.points) < 30:
        raise RuntimeError("Too few target depth points after voxel downsampling")
    result.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel * 4.0, max_nn=40))
    return result


def registration_quality(source, target, distance, transform):
    result = o3d.pipelines.registration.evaluate_registration(
        source, target, distance, transform)
    return float(result.fitness), float(result.inlier_rmse)


def registration_score(fitness, rmse, distance):
    if fitness <= 0:
        return np.inf
    return rmse + (1.0 - fitness) * distance


def center_error(source, target, transform):
    source_center = np.asarray(source.points).mean(axis=0)
    target_center = np.asarray(target.points).mean(axis=0)
    transformed = transform[:3, :3] @ source_center + transform[:3, 3]
    return float(np.linalg.norm(transformed - target_center))


def register_pair(source, target, voxel):
    """Register slowly changing views without letting coarse plane ICP diverge."""
    transform = np.eye(4)
    transform[:3, 3] = (np.asarray(target.points).mean(axis=0) -
                        np.asarray(source.points).mean(axis=0))
    method = "centroid"

    # Sparse normals on a round object make coarse point-to-plane ICP singular.
    # Establish correspondence with point-to-point first, then consider plane
    # refinement only at full resolution and only when it actually scores better.
    for scale_voxel, iterations in ((voxel * 4.0, 50), (voxel * 2.0, 40)):
        source_level = prepare(source, scale_voxel)
        target_level = prepare(target, scale_voxel)
        distance = scale_voxel * 2.5
        baseline = registration_quality(source_level, target_level, distance, transform)
        candidate = o3d.pipelines.registration.registration_icp(
            source_level, target_level, distance, transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations))
        candidate_quality = (float(candidate.fitness), float(candidate.inlier_rmse))
        if (registration_score(*candidate_quality, distance) <= (
                registration_score(*baseline, distance) * 1.05) and
                center_error(source_level, target_level, candidate.transformation) <=
                distance * 2.0):
            transform = candidate.transformation
            method = "point-to-point"

    source_fine = prepare(source, voxel)
    target_fine = prepare(target, voxel)
    distance = voxel * 2.5
    candidates = [(transform, method)]
    for estimation, name in (
            (o3d.pipelines.registration.TransformationEstimationPointToPoint(),
             "point-to-point"),
            (o3d.pipelines.registration.TransformationEstimationPointToPlane(),
             "point-to-plane")):
        result = o3d.pipelines.registration.registration_icp(
            source_fine, target_fine, distance, transform, estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        if center_error(source_fine, target_fine, result.transformation) <= distance * 2.0:
            candidates.append((result.transformation, name))

    ranked = []
    for candidate, name in candidates:
        fitness, rmse = registration_quality(source_fine, target_fine, distance, candidate)
        ranked.append((registration_score(fitness, rmse, distance), candidate,
                       name, fitness, rmse))
    _, transform, method, fitness, rmse = min(ranked, key=lambda item: item[0])
    return transform, fitness, rmse, method


def retain_largest_component(mesh):
    clusters, counts, _ = mesh.cluster_connected_triangles()
    if not counts:
        raise RuntimeError("Poisson reconstruction produced no triangle component")
    clusters = np.asarray(clusters)
    largest = int(np.argmax(np.asarray(counts)))
    mesh.remove_triangles_by_mask(clusters != largest)
    mesh.remove_unreferenced_vertices()
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--voxel", type=float, default=0.003)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--min-depth", type=float, default=0.15)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--max-surface-depth-span", type=float, default=0.2)
    parser.add_argument("--fitness-min", type=float, default=0.3)
    parser.add_argument("--rmse-max", type=float, default=0.015)
    parser.add_argument("--poisson-depth", type=int, default=8)
    parser.add_argument("--max-triangles", type=int, default=100000)
    args = parser.parse_args()
    if args.voxel <= 0 or args.stride < 1 or not 5 <= args.poisson_depth <= 11:
        parser.error("invalid voxel, stride, or poisson depth")

    sequence = Path(args.sequence).resolve()
    metadata = json.loads((sequence / "meta.json").read_text(encoding="utf-8"))
    K = np.asarray(metadata["K"], dtype=np.float64)
    rgb_paths = sorted((sequence / "rgb").glob("*.png"))
    if len(rgb_paths) < 3:
        raise RuntimeError("At least three segmented RGB-D views are required")

    clouds = []
    for rgb_path in rgb_paths:
        depth_path = sequence / "depth" / f"{rgb_path.stem}.npy"
        mask_path = sequence / "mask" / f"{rgb_path.stem}.png"
        if not depth_path.is_file() or not mask_path.is_file():
            raise RuntimeError(f"Missing depth or mask for frame {rgb_path.stem}")
        cloud = load_cloud(rgb_path, depth_path, mask_path, K, args.stride,
                           args.min_depth, args.max_depth, args.max_surface_depth_span)
        if len(cloud.points) < 100:
            raise RuntimeError(f"Frame {rgb_path.stem} has only {len(cloud.points)} target points")
        clouds.append(cloud)

    current_to_reference = [np.eye(4)]
    registrations = []
    for index in range(1, len(clouds)):
        current_to_previous, fitness, rmse, method = register_pair(
            clouds[index], clouds[index - 1], args.voxel)
        if fitness < args.fitness_min or rmse > args.rmse_max:
            raise RuntimeError(
                f"ICP quality failed at frame {index}: fitness={fitness:.3f}, "
                f"rmse={rmse:.4f} m; rotate more slowly and keep the camera fixed")
        current_to_reference.append(current_to_reference[-1] @ current_to_previous)
        registrations.append({"frame": rgb_paths[index].stem, "fitness": fitness,
                              "rmse_m": rmse, "method": method})
        print(f"ICP {index + 1}/{len(clouds)} fitness={fitness:.3f} "
              f"rmse={rmse:.4f} m method={method}", flush=True)

    transformed = []
    for cloud, transform in zip(clouds, current_to_reference):
        aligned = copy.deepcopy(cloud)
        aligned.transform(transform)
        transformed.append(aligned)
    fused_reference = transformed[0]
    for cloud in transformed[1:]:
        fused_reference += cloud
    fused_reference = fused_reference.voxel_down_sample(args.voxel)
    fused_reference, _ = fused_reference.remove_statistical_outlier(
        nb_neighbors=20, std_ratio=2.0)
    if len(fused_reference.points) < 500:
        raise RuntimeError("Fused reference cloud is too small for mesh reconstruction")

    reference_points = np.asarray(fused_reference.points)
    center = np.median(reference_points, axis=0)
    centered = copy.deepcopy(fused_reference)
    centered.translate(-center)
    centered.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(
        radius=args.voxel * 4.0, max_nn=50))
    try:
        centered.orient_normals_consistent_tangent_plane(30)
    except RuntimeError:
        centered.orient_normals_towards_camera_location(np.zeros(3))

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        centered, depth=args.poisson_depth)
    densities = np.asarray(densities)
    if len(densities):
        mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))
    bounds = centered.get_axis_aligned_bounding_box()
    bounds = o3d.geometry.AxisAlignedBoundingBox(
        bounds.min_bound - args.voxel * 2, bounds.max_bound + args.voxel * 2)
    mesh = mesh.crop(bounds)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh = retain_largest_component(mesh)
    if len(mesh.triangles) > args.max_triangles:
        mesh = mesh.simplify_quadric_decimation(args.max_triangles)
    mesh.compute_vertex_normals()

    extent = mesh.get_axis_aligned_bounding_box().get_extent()
    if len(mesh.vertices) < 100 or len(mesh.triangles) < 100 or np.any(extent < 0.005) or np.any(extent > 2.0):
        raise RuntimeError(
            f"Reconstructed mesh failed size/topology validation: extent={extent.tolist()}, "
            f"vertices={len(mesh.vertices)}, triangles={len(mesh.triangles)}")

    reconstruction = sequence / "reconstruction"
    pose_dir = reconstruction / "cam_in_ob"
    pose_dir.mkdir(parents=True, exist_ok=True)
    object_to_reference = np.eye(4)
    object_to_reference[:3, 3] = center
    for rgb_path, transform in zip(rgb_paths, current_to_reference):
        cam_in_ob = np.linalg.inv(transform) @ object_to_reference
        np.savetxt(pose_dir / f"{rgb_path.stem}.txt", cam_in_ob, fmt="%.9g")
    o3d.io.write_point_cloud(str(reconstruction / "reference_cloud.ply"), centered,
                             write_ascii=False, compressed=False)
    if not o3d.io.write_triangle_mesh(str(reconstruction / "model.obj"), mesh,
                                      write_vertex_normals=True):
        raise RuntimeError("Open3D failed to write reconstructed model.obj")

    report = {
        "frames": len(clouds),
        "reference_points": len(centered.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "mesh_extent_m": extent.tolist(),
        "mean_icp_fitness": float(np.mean([item["fitness"] for item in registrations])),
        "max_icp_rmse_m": float(max(item["rmse_m"] for item in registrations)),
        "registrations": registrations,
    }
    (reconstruction / "quality.json").write_text(json.dumps(report, indent=2),
                                                   encoding="utf-8")
    print("RECONSTRUCTION " + json.dumps({key: value for key, value in report.items()
                                           if key != "registrations"}))


if __name__ == "__main__":
    main()
