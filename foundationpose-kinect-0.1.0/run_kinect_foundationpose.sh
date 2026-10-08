#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
original_args=("$@")
cpp="$root/foundationpose_cpp"
init_dir="$cpp/test_data/live_init"
output_root="$cpp/test_output/live"
config="$root/configs/kinect_foundationpose.json"
conda_env="${FP_CONDA_ENV:-kinect}"
object_override=""
everything=0
prompt_override=""
segmentation_mode_override=""
port_override=""
max_frames_override=""
warmup_override=""
layout_override=""
pointcloud_override=""
hybrid_override=""
hybrid_sam_hz_override=""
hybrid_window_override=""
image_override=""
replay=""
headless_override=""
skip_initialization=0
container=""
gpu_monitor_pid=""
cleanup_done=0

usage() {
  cat <<'EOF'
Usage: ./run_kinect_foundationpose.sh [options]

  --config FILE              Pipeline/object JSON configuration
  --object NAME              Select an object from the configuration
  --everything               Initialize a clicked target mask without any object mesh
  --segmentation-mode MODE   text or click
  --click                    Shortcut for --segmentation-mode click
  --headless                 Disable the OpenCV preview
  --max-frames N             Stop after N tracking frames (0 is unlimited)
  --replay DIR               Replay a recording without opening a camera
  --skip-initialization      Reuse live_init RGB-D, mask, and intrinsics
  --prompt TEXT              Comma-separated SAM3 prompt fallback chain
  --visualization LAYOUT     rgb, rgbd, or rgbd_cloud
  --pointcloud               Enable point-cloud processing/export
  --no-pointcloud            Disable point-cloud processing/export
  --hybrid                   Enable asynchronous SAM3 + FoundationPose recovery
  --no-hybrid                Disable hybrid mode even when enabled in config
  --sam-hz HZ                Hybrid SAM3 target rate (default: 5 Hz)
  --sam-window-seconds SEC    Bounded hybrid frame/mask history (default: 2 s)
  --port N                   Local TCP port (default: 50051)
  --warmup N                 Kinect frames discarded after each open
  --image NAME               FoundationPose Docker image name
  -h, --help                 Show this help
EOF
}

need_value() {
  if (($# < 2)); then
    echo "ERROR: $1 requires a value" >&2
    exit 2
  fi
}

while (($#)); do
  case "$1" in
    --config) need_value "$@"; config="$2"; shift 2 ;;
    --object) need_value "$@"; object_override="$2"; shift 2 ;;
    --everything) everything=1; shift ;;
    --segmentation-mode) need_value "$@"; segmentation_mode_override="$2"; shift 2 ;;
    --click) segmentation_mode_override=click; shift ;;
    --headless) headless_override=1; shift ;;
    --skip-initialization) skip_initialization=1; shift ;;
    --max-frames) need_value "$@"; max_frames_override="$2"; shift 2 ;;
    --replay) need_value "$@"; replay="$2"; shift 2 ;;
    --prompt) need_value "$@"; prompt_override="$2"; shift 2 ;;
    --visualization) need_value "$@"; layout_override="$2"; shift 2 ;;
    --pointcloud) pointcloud_override=1; shift ;;
    --no-pointcloud) pointcloud_override=0; shift ;;
    --hybrid) hybrid_override=1; shift ;;
    --no-hybrid) hybrid_override=0; shift ;;
    --sam-hz) need_value "$@"; hybrid_sam_hz_override="$2"; shift 2 ;;
    --sam-window-seconds) need_value "$@"; hybrid_window_override="$2"; shift 2 ;;
    --port) need_value "$@"; port_override="$2"; shift 2 ;;
    --warmup) need_value "$@"; warmup_override="$2"; shift 2 ;;
    --image) need_value "$@"; image_override="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$config" != /* ]]; then
  config="$root/$config"
fi
config_args=(--config "$config" --root "$root" --shell)
[[ -z "$object_override" ]] || config_args+=(--object "$object_override")
((everything == 0)) || config_args+=(--everything)
config_shell="$(/usr/bin/python3 "$root/tools/resolve_kinect_config.py" "${config_args[@]}")"
eval "$config_shell"
object_name="$cfg_object_name"
mesh="$cfg_mesh"
prompt="${prompt_override:-$cfg_prompt}"
segmentation_mode="${segmentation_mode_override:-$cfg_segmentation_mode}"
if ((everything)); then
  if [[ -n "$segmentation_mode_override" && "$segmentation_mode_override" != click ]]; then
    echo "ERROR: --everything initialization requires click segmentation" >&2
    exit 2
  fi
  segmentation_mode=click
fi
port="${port_override:-$cfg_port}"
max_frames="${max_frames_override:-$cfg_max_frames}"
warmup="${warmup_override:-$cfg_warmup}"
headless="${headless_override:-$cfg_headless}"
visualization_layout="${layout_override:-$cfg_visualization_layout}"
pointcloud_enabled="${pointcloud_override:-$cfg_pointcloud_enabled}"
hybrid_enabled="${hybrid_override:-$cfg_hybrid_enabled}"
hybrid_sam_hz="${hybrid_sam_hz_override:-$cfg_hybrid_sam_hz}"
hybrid_window_seconds="${hybrid_window_override:-$cfg_hybrid_window_seconds}"
image="${image_override:-${FP_IMAGE:-$cfg_image}}"
stage_count=$((everything ? 2 : 4))

for value in "$port" "$max_frames" "$warmup"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "ERROR: numeric options must be non-negative integers" >&2
    exit 2
  fi
done
if ((port < 1 || port > 65535)); then
  echo "ERROR: port must be between 1 and 65535" >&2
  exit 2
fi
if [[ ! "$hybrid_enabled" =~ ^[01]$ ]]; then
  echo "ERROR: hybrid mode must resolve to 0 or 1" >&2
  exit 2
fi
if [[ ! "$hybrid_sam_hz" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] ||
    [[ ! "$hybrid_window_seconds" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
  echo "ERROR: --sam-hz and --sam-window-seconds must be positive numbers" >&2
  exit 2
fi
if ! awk -v hz="$hybrid_sam_hz" -v window="$hybrid_window_seconds" \
    'BEGIN {exit !(hz > 0 && hz <= 30 && window >= 0.2 && window <= 60)}'; then
  echo "ERROR: SAM Hz must be in (0,30] and window must be in [0.2,60] seconds" >&2
  exit 2
fi
if [[ "$segmentation_mode" != text && "$segmentation_mode" != click ]]; then
  echo "ERROR: segmentation mode must be text or click" >&2
  exit 2
fi
if [[ "$visualization_layout" != rgb && "$visualization_layout" != rgbd &&
      "$visualization_layout" != rgbd_cloud ]]; then
  echo "ERROR: visualization layout must be rgb, rgbd, or rgbd_cloud" >&2
  exit 2
fi
if ((pointcloud_enabled == 0)) && [[ "$visualization_layout" == rgbd_cloud ]]; then
  echo "Point cloud is disabled; using rgbd visualization."
  visualization_layout=rgbd
fi
if [[ "$segmentation_mode" == click && -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  echo "ERROR: click segmentation requires a graphical desktop session" >&2
  exit 1
fi

if [[ -n "$replay" && "$replay" != /* ]]; then
  replay="$root/$replay"
fi
if [[ -n "$replay" && ! -d "$replay" ]]; then
  echo "ERROR: replay directory does not exist: $replay" >&2
  exit 1
fi

# A newly-added docker group is not visible in an existing login shell.
# Re-enter the script once with that supplementary group active.
if ((everything == 0)) && command -v docker >/dev/null 2>&1 &&
    ! docker info >/dev/null 2>&1 &&
    [[ "${FP_DOCKER_GROUP_REEXEC:-0}" != 1 ]]; then
  docker_members="$(getent group docker 2>/dev/null | cut -d: -f4)"
  current_user="$(id -un)"
  if [[ ",$docker_members," == *",$current_user,"* ]]; then
    printf -v reexec_command '%q ' env FP_DOCKER_GROUP_REEXEC=1 "$0" "${original_args[@]}"
    echo "Docker group membership is not active in this shell; re-entering with group docker."
    exec sg docker -c "$reexec_command"
  fi
fi

run_id="run_$(date +%Y%m%d_%H%M%S)"
run_output="$output_root/$run_id"
stage_file="$run_output/gpu_stage"
gpu_csv="$run_output/gpu_memory.csv"
server_log="$run_output/foundationpose.log"
mkdir -p "$init_dir" "$cpp/test_data/live_object" "$run_output"
cp "$config" "$run_output/config.json"
cat > "$run_output/runtime_settings.txt" <<EOF
object=$object_name
mesh=$mesh
segmentation_mode=$segmentation_mode
prompt=$prompt
port=$port
max_frames=$max_frames
visualization=$visualization_layout
pointcloud_enabled=$pointcloud_enabled
everything=$everything
pointcloud_mode=$cfg_pointcloud_mode
hybrid_enabled=$hybrid_enabled
hybrid_sam_hz=$hybrid_sam_hz
hybrid_window_seconds=$hybrid_window_seconds
EOF

set_stage() {
  printf '%s\n' "$1" > "$stage_file"
}

monitor_gpu() {
  printf 'timestamp,stage,gpu_used_mib\n' > "$gpu_csv"
  while true; do
    local stage used
    stage="$(cat "$stage_file" 2>/dev/null || printf unknown)"
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | sed -n '1p' | tr -d ' ')"
    printf '%s,%s,%s\n' "$(date --iso-8601=seconds)" "$stage" "${used:-NA}" >> "$gpu_csv"
    sleep 1
  done
}

cleanup() {
  local status=$?
  if ((cleanup_done)); then
    return
  fi
  cleanup_done=1
  set +e
  set_stage cleanup
  if [[ -n "$gpu_monitor_pid" ]]; then
    kill "$gpu_monitor_pid" 2>/dev/null
    wait "$gpu_monitor_pid" 2>/dev/null
  fi
  if [[ -n "$container" ]]; then
    docker logs "$container" > "$server_log" 2>&1
    docker rm -f "$container" >/dev/null 2>&1
  fi
  if [[ -s "$gpu_csv" ]]; then
    awk -F, 'NR > 1 && $3 ~ /^[0-9]+$/ {if ($3 > max) max=$3} END {if (max) print "peak_gpu_memory_mib=" max}' \
      "$gpu_csv" > "$run_output/gpu_peak.txt"
  fi
  rm -f "$stage_file"
  if ((status != 0)); then
    echo "FAILED (exit $status). Artifacts retained in $run_output" >&2
  fi
  trap - EXIT
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

required_commands=(conda nvidia-smi)
((everything != 0)) || required_commands+=(docker)
for command in "${required_commands[@]}"; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "ERROR: required command is missing: $command" >&2
    echo "Run: sudo $root/tools/setup_ubuntu22.sh --install" >&2
    exit 1
  fi
done
if ((everything == 0)) && ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is unavailable to this user; check the daemon and docker group" >&2
  exit 1
fi
if ! nvidia-smi -L >/dev/null; then
  echo "ERROR: the NVIDIA driver does not expose a GPU" >&2
  exit 1
fi

if ((everything == 0)); then
  cp "$mesh" "$cpp/test_data/live_object/mesh.obj"
fi
echo "Configuration: mode=$([[ $everything == 1 ]] && printf everything || printf foundationpose) object=$object_name segmentation=$segmentation_mode visualization=$visualization_layout hybrid=$hybrid_enabled"
set_stage idle
monitor_gpu &
gpu_monitor_pid=$!

kinect_prefix="$(conda run --no-capture-output -n "$conda_env" python -c 'import sys; print(sys.prefix)')"
kinect_libstdcpp="$kinect_prefix/lib/libstdc++.so.6"
if [[ ! -f "$kinect_libstdcpp" ]]; then
  echo "ERROR: $conda_env libstdc++ is missing: $kinect_libstdcpp" >&2
  exit 1
fi
kinect_preload="$kinect_libstdcpp${LD_PRELOAD:+:$LD_PRELOAD}"
kinect=(conda run --no-capture-output -n "$conda_env" env LD_PRELOAD="$kinect_preload"
  python "$root/kinect_foundationpose_live.py")
sam_prefix="$kinect_prefix"
sam_libstdcpp="$sam_prefix/lib/libstdc++.so.6"
if [[ ! -f "$sam_libstdcpp" ]]; then
  echo "ERROR: $conda_env libstdc++ is missing: $sam_libstdcpp" >&2
  exit 1
fi
sam_preload="$sam_libstdcpp${LD_PRELOAD:+:$LD_PRELOAD}"
if ((skip_initialization == 0)); then
  set_stage capture_initial
  echo "[1/$stage_count] Capture initialization RGB-D and close the camera"
  capture_args=(capture --output "$init_dir" --warmup "$warmup")
  [[ -z "$replay" ]] || capture_args+=(--replay "$replay")
  "${kinect[@]}" "${capture_args[@]}"
  rm -f "$init_dir/mask.png" "$init_dir/mask_overlay.png" "$init_dir/object.txt"

  set_stage sam3
  echo "[2/$stage_count] Run SAM3 once, then release its GPU memory"
  conda run --no-capture-output -n "$conda_env" env -u PYTHONPATH \
    QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}" LD_PRELOAD="$sam_preload" \
    python "$root/server/sam3_segment.py" \
    --input "$init_dir/rgb.png" --output "$init_dir/mask.png" \
    --ckpt "$cfg_sam_checkpoint" --bpe "$cfg_sam_bpe" \
    --mode "$segmentation_mode" --prompts "$prompt" --thr "$cfg_sam_threshold"
  printf '%s\n' "$object_name" > "$init_dir/object.txt"
else
  echo "[1/$stage_count] Reuse initialization RGB-D and mask"
  echo "[2/$stage_count] SAM3 skipped"
  if [[ -s "$init_dir/object.txt" && "$(<"$init_dir/object.txt")" != "$object_name" ]]; then
    echo "ERROR: saved mask belongs to object '$(<"$init_dir/object.txt")', not '$object_name'; rerun without --skip-initialization" >&2
    exit 1
  fi
fi

"${kinect[@]}" validate --init-dir "$init_dir"
if pgrep -u "$(id -u)" -f '[s]am3_segment.py' >/dev/null; then
  echo "ERROR: initialization SAM3 process is still alive" >&2
  exit 1
fi
set_stage sam_released
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader > "$run_output/after_sam_processes.txt" 2>&1 || true

if ((everything)); then
  set_stage complete
  echo "Everything initialization completed. RGB-D and clicked mask: $init_dir"
  echo "No object mesh was resolved and no FoundationPose container was started."
  exit 0
fi

if ! docker image inspect "$image" >/dev/null 2>&1; then
  echo "[3/4] Build pinned Ubuntu 22.04 / TensorRT 10 image"
  FP_IMAGE="$image" "$cpp/tools/build_ubuntu_image.sh"
else
  echo "[3/4] Use Docker image $image"
fi

container="foundationpose-live-$$"
set_stage foundationpose_startup
cpp_pointcloud_save_every="$cfg_pointcloud_save_every"
((pointcloud_enabled != 0)) || cpp_pointcloud_save_every=0
docker run -d --name "$container" --gpus all \
  -p "127.0.0.1:${port}:${port}" \
  -v "$cpp:/workspace" -w /workspace "$image" \
  bash /workspace/tools/run_live_fp.sh "$port" "$object_name" \
    "/workspace/test_output/live/$run_id/pointcloud" "$cpp_pointcloud_save_every" \
    "$cfg_pointcloud_stride" "$cfg_pointcloud_min_depth" "$cfg_pointcloud_max_depth" \
    "$cfg_pointcloud_max_points" "/workspace/test_output/live/$run_id/visualization" \
    "$cfg_visualization_save_every" "$cfg_pointcloud_crop_margin" \
    "$cfg_pointcloud_reference_frames" "$cfg_pointcloud_reference_voxel" \
    "$cfg_pointcloud_mesh_points" >/dev/null

ready=0
for _ in $(seq 1 900); do
  recent_logs="$(docker logs --tail 80 "$container" 2>&1 || true)"
  if grep -q "READY port=$port" <<< "$recent_logs"; then
    ready=1
    break
  fi
  if [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null)" != true ]]; then
    break
  fi
  sleep 2
done
if ((ready == 0)); then
  docker logs "$container" > "$server_log" 2>&1 || true
  echo "ERROR: FoundationPose did not become READY; inspect $server_log" >&2
  exit 1
fi

set_stage foundationpose
echo "[4/4] Register the saved frame, reopen Kinect, then Track"
stream_args=(stream --init-dir "$init_dir" --mesh "$mesh" --port "$port"
  --warmup "$warmup" --max-frames "$max_frames"
  --pose-output "$run_output/latest_pose.txt"
  --metrics-output "$run_output/metrics.json"
  --visualization-layout "$visualization_layout"
  --preview-scale "$cfg_preview_scale"
  --pointcloud-output "$run_output/pointcloud"
  --pointcloud-mode "$cfg_pointcloud_mode"
  --pointcloud-save-every "$cfg_pointcloud_save_every"
  --pointcloud-preview-every "$cfg_pointcloud_preview_every"
  --pointcloud-stride "$cfg_pointcloud_stride"
  --pointcloud-min-depth "$cfg_pointcloud_min_depth"
  --pointcloud-max-depth "$cfg_pointcloud_max_depth"
  --pointcloud-max-points "$cfg_pointcloud_max_points"
  --pointcloud-mesh-points "$cfg_pointcloud_mesh_points"
  --pointcloud-reference-frames "$cfg_pointcloud_reference_frames"
  --pointcloud-reference-voxel "$cfg_pointcloud_reference_voxel"
  --pointcloud-crop-margin "$cfg_pointcloud_crop_margin"
  --pointcloud-yaw "$cfg_pointcloud_yaw"
  --pointcloud-pitch "$cfg_pointcloud_pitch")
((pointcloud_enabled == 0)) || stream_args+=(--pointcloud-enabled)
if ((hybrid_enabled != 0)); then
  stream_args+=(--hybrid-enabled
    --hybrid-sam-hz "$hybrid_sam_hz"
    --hybrid-window-seconds "$hybrid_window_seconds"
    --hybrid-reregister-center-px "$cfg_hybrid_reregister_center_px"
    --hybrid-reregister-cooldown "$cfg_hybrid_reregister_cooldown"
    --hybrid-min-mask-pixels "$cfg_hybrid_min_mask_pixels"
    --hybrid-max-area-change "$cfg_hybrid_max_area_change"
    --sam-checkpoint "$cfg_sam_checkpoint"
    --sam-bpe "$cfg_sam_bpe"
    --sam-prompts "$prompt"
    --sam-threshold "$cfg_sam_threshold")
fi
((headless == 0)) || stream_args+=(--headless)
[[ -z "$replay" ]] || stream_args+=(--replay "$replay")
"${kinect[@]}" "${stream_args[@]}"

echo "Completed. Metrics and logs: $run_output"
