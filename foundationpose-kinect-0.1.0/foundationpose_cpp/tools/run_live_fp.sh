#!/usr/bin/env bash
set -euo pipefail

cd /workspace
port="${1:-50051}"
object="${2:-object}"
pointcloud_dir="${3:-}"
pointcloud_save_every="${4:-0}"
pointcloud_stride="${5:-3}"
pointcloud_min_depth="${6:-0.15}"
pointcloud_max_depth="${7:-2.0}"
pointcloud_max_points="${8:-120000}"
visualization_dir="${9:-}"
visualization_save_every="${10:-0}"
pointcloud_crop_margin="${11:-0.005}"
pointcloud_reference_frames="${12:-5}"
pointcloud_reference_voxel="${13:-0.003}"
pointcloud_mesh_points="${14:-12000}"

/workspace/tools/ensure_local_engines.sh

cmake -S . -B build -DENABLE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "$(nproc)" --target foundationpose_live

exec ./build/bin/foundationpose_live --port "$port" \
  --refiner /workspace/models/refiner_hwc_dynamic_fp16.engine \
  --scorer /workspace/models/scorer_hwc_dynamic_fp16.engine \
  --mesh /workspace/test_data/live_object/mesh.obj \
  --object "$object" \
  --pointcloud-dir "$pointcloud_dir" \
  --pointcloud-save-every "$pointcloud_save_every" \
  --pointcloud-stride "$pointcloud_stride" \
  --pointcloud-min-depth "$pointcloud_min_depth" \
  --pointcloud-max-depth "$pointcloud_max_depth" \
  --pointcloud-max-points "$pointcloud_max_points" \
  --visualization-dir "$visualization_dir" \
  --visualization-save-every "$visualization_save_every" \
  --pointcloud-crop-margin "$pointcloud_crop_margin" \
  --pointcloud-reference-frames "$pointcloud_reference_frames" \
  --pointcloud-reference-voxel "$pointcloud_reference_voxel" \
  --pointcloud-mesh-points "$pointcloud_mesh_points" \
  --output /workspace/test_output/live
