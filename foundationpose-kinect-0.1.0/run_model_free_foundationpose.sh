#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config="$root/configs/kinect_foundationpose.json"
conda_env="${FP_CONDA_ENV:-kinect}"
name=""
prompt=""
mode_override=""
replay=""
resume_sequence=""
headless=0
build_only=0
replace=0
capture_frames_override=""
capture_every_override=""

usage() {
  cat <<'EOF'
Usage: ./run_model_free_foundationpose.sh --name NAME [options]

  --name NAME                 Configuration key for the reconstructed object
  --prompt TEXT               Comma-separated SAM3 prompts (defaults to NAME)
  --segmentation-mode MODE    text or click; click is interactive on frame 0
  --click                     Shortcut for --segmentation-mode click
  --config FILE               Pipeline JSON configuration
  --replay DIR                Use an RGB-D recording instead of Azure Kinect
  --resume-sequence DIR       Reuse existing RGB/depth/mask and restart reconstruction
  --capture-frames N          Number of saved reconstruction views
  --capture-every N           Save every Nth Kinect frame
  --headless                  Disable capture preview
  --build-only                Register the mesh but do not start live tracking
  --replace                   Replace an existing object entry of the same name
  -h, --help                  Show this help

Keep the camera fixed. Slowly rotate the object once, tilt it, then rotate it a
second time. SAM3 exits before Open3D starts; FoundationPose starts only after
reconstruction has completed.
EOF
}

need_value() {
  if (($# < 2)); then echo "ERROR: $1 requires a value" >&2; exit 2; fi
}

while (($#)); do
  case "$1" in
    --name) need_value "$@"; name="$2"; shift 2 ;;
    --prompt) need_value "$@"; prompt="$2"; shift 2 ;;
    --segmentation-mode) need_value "$@"; mode_override="$2"; shift 2 ;;
    --click) mode_override=click; shift ;;
    --config) need_value "$@"; config="$2"; shift 2 ;;
    --replay) need_value "$@"; replay="$2"; shift 2 ;;
    --resume-sequence) need_value "$@"; resume_sequence="$2"; shift 2 ;;
    --capture-frames) need_value "$@"; capture_frames_override="$2"; shift 2 ;;
    --capture-every) need_value "$@"; capture_every_override="$2"; shift 2 ;;
    --headless) headless=1; shift ;;
    --build-only) build_only=1; shift ;;
    --replace) replace=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$name" ]] || { echo "ERROR: --name is required" >&2; usage >&2; exit 2; }
[[ "$config" == /* ]] || config="$root/$config"
[[ -z "$replay" || "$replay" == /* ]] || replay="$root/$replay"
[[ -z "$resume_sequence" || "$resume_sequence" == /* ]] || resume_sequence="$root/$resume_sequence"
[[ -z "$replay" || -d "$replay" ]] || { echo "ERROR: replay not found: $replay" >&2; exit 1; }
[[ -z "$resume_sequence" || -d "$resume_sequence" ]] || {
  echo "ERROR: sequence not found: $resume_sequence" >&2; exit 1; }
[[ -z "$resume_sequence" || -z "$replay" ]] || {
  echo "ERROR: --resume-sequence and --replay cannot be used together" >&2; exit 2; }

config_shell="$(/usr/bin/python3 "$root/tools/resolve_kinect_config.py" \
  --config "$config" --root "$root" --allow-missing-mesh --shell)"
eval "$config_shell"
mode="${mode_override:-$cfg_segmentation_mode}"
prompt="${prompt:-$name}"
capture_frames="${capture_frames_override:-$cfg_model_free_capture_frames}"
capture_every="${capture_every_override:-$cfg_model_free_capture_every}"
[[ "$mode" == text || "$mode" == click ]] || { echo "ERROR: invalid segmentation mode" >&2; exit 2; }
[[ "$capture_frames" =~ ^[0-9]+$ && "$capture_every" =~ ^[0-9]+$ ]] || {
  echo "ERROR: capture values must be integers" >&2; exit 2; }
((capture_frames >= 3 && capture_every >= 1)) || {
  echo "ERROR: capture-frames >= 3 and capture-every >= 1 are required" >&2; exit 2; }
if [[ "$mode" == click && -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  echo "ERROR: click mode requires a graphical desktop session" >&2
  exit 1
fi
if ((replace == 0)) && /usr/bin/python3 - "$config" "$name" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if sys.argv[2] in config.get("objects", {}) else 1)
PY
then
  echo "ERROR: object '$name' already exists; choose another name or pass --replace" >&2
  exit 1
fi

if [[ -n "$resume_sequence" ]]; then
  sequence="$resume_sequence"
  for required in meta.json rgb depth mask; do
    [[ -e "$sequence/$required" ]] || {
      echo "ERROR: resumed sequence is missing $required: $sequence" >&2; exit 1; }
  done
else
  run_id="run_$(date +%Y%m%d_%H%M%S)"
  sequence="$cfg_model_free_root/$name/$run_id"
  mkdir -p "$sequence"
fi
echo "Model-free workspace: $sequence"

if [[ -z "$resume_sequence" ]]; then
  kinect_prefix="$(conda run --no-capture-output -n "$conda_env" python -c 'import sys; print(sys.prefix)')"
  kinect_preload="$kinect_prefix/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  kinect=(conda run --no-capture-output -n "$conda_env" env LD_PRELOAD="$kinect_preload"
    python "$root/kinect_foundationpose_live.py")

  echo "[1/4] Capture rotating-object RGB-D sequence"
  capture_args=(capture-sequence --output "$sequence" --warmup "$cfg_warmup"
    --max-frames "$capture_frames" --frame-step "$capture_every")
  [[ -z "$replay" ]] || capture_args+=(--replay "$replay")
  ((headless == 0)) || capture_args+=(--headless)
  "${kinect[@]}" "${capture_args[@]}"

  sam_prefix="$kinect_prefix"
  sam_preload="$sam_prefix/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  echo "[2/4] Segment all views with one SAM3 model load"
  conda run --no-capture-output -n "$conda_env" env -u PYTHONPATH \
    QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}" LD_PRELOAD="$sam_preload" \
    python "$root/server/sam3_batch_segment.py" --sequence "$sequence" \
    --ckpt "$cfg_sam_checkpoint" --bpe "$cfg_sam_bpe" --mode "$mode" \
    --prompts "$prompt" --threshold "$cfg_sam_threshold" \
    --area-change-ratio "$cfg_model_free_area_change"
  if pgrep -u "$(id -u)" -f '[s]am3_batch_segment.py' >/dev/null; then
    echo "ERROR: SAM3 process did not exit; reconstruction was not started" >&2
    exit 1
  fi
else
  echo "[1/4] Reuse captured RGB-D sequence"
  echo "[2/4] Verify existing masks and resume SAM3 at the first missing frame"
  sam_prefix="$(conda run --no-capture-output -n "$conda_env" python -c 'import sys; print(sys.prefix)')"
  sam_preload="$sam_prefix/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
  conda run --no-capture-output -n "$conda_env" env -u PYTHONPATH \
    QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}" LD_PRELOAD="$sam_preload" \
    python "$root/server/sam3_batch_segment.py" --sequence "$sequence" \
    --ckpt "$cfg_sam_checkpoint" --bpe "$cfg_sam_bpe" --mode "$mode" \
    --prompts "$prompt" --threshold "$cfg_sam_threshold" \
    --area-change-ratio "$cfg_model_free_area_change" --resume
  if pgrep -u "$(id -u)" -f '[s]am3_batch_segment.py' >/dev/null; then
    echo "ERROR: SAM3 process did not exit; reconstruction was not started" >&2
    exit 1
  fi
fi

echo "[3/4] Estimate object motion, fuse reference cloud, and reconstruct mesh"
reconstruct_env="$conda_env"
if ! conda run -n "$reconstruct_env" python -c 'import cv2, open3d' >/dev/null 2>&1; then
  echo "ERROR: Open3D is unavailable in conda environment '$conda_env'; run:" >&2
  echo "  conda env create -f $root/environments/kinect.yml" >&2
  exit 1
fi
conda run --no-capture-output -n "$reconstruct_env" env -u PYTHONPATH \
  python "$root/tools/reconstruct_model_free.py" --sequence "$sequence" \
  --voxel "$cfg_model_free_voxel" --fitness-min "$cfg_model_free_icp_fitness" \
  --rmse-max "$cfg_model_free_icp_rmse" --poisson-depth "$cfg_model_free_poisson_depth" \
  --max-surface-depth-span "$cfg_model_free_surface_depth_span" \
  --max-triangles "$cfg_model_free_max_triangles"

register_args=(--config "$config" --root "$root" --name "$name"
  --mesh "$sequence/reconstruction/model.obj" --prompts "$prompt")
((replace == 0)) || register_args+=(--replace)
/usr/bin/python3 "$root/tools/register_reconstructed_object.py" "${register_args[@]}"
echo "[4/4] Reconstructed object is available as '$name'"

if ((build_only)); then
  echo "Build complete. Quality report: $sequence/reconstruction/quality.json"
  exit 0
fi

live_args=(--config "$config" --object "$name" --segmentation-mode "$mode" --prompt "$prompt")
((headless == 0)) || live_args+=(--headless)
[[ -z "$replay" ]] || live_args+=(--replay "$replay")
exec "$root/run_kinect_foundationpose.sh" "${live_args[@]}"
