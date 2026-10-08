#!/usr/bin/env bash
set -euo pipefail

cd /workspace
trtexec="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [[ ! -x "$trtexec" ]]; then
  echo "ERROR: TensorRT trtexec is missing at $trtexec" >&2
  exit 1
fi

gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '1p')"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
trt_version="$(dpkg-query -W -f='${Version}' libnvinfer10 2>/dev/null)"
if [[ -z "$trt_version" ]]; then
  echo "ERROR: unable to determine the installed TensorRT version" >&2
  exit 1
fi
fingerprint="$(printf 'gpu_uuid=%s\ngpu_name=%s\ntrt=%s\nprecision=fp16' \
  "$gpu_uuid" "$gpu_name" "$trt_version")"

ensure_engine() {
  local stem="$1"
  local onnx="models/${stem}_hwc.onnx"
  local engine="models/${stem}_hwc_dynamic_fp16.engine"
  local metadata="${engine}.metadata"
  local temporary="${engine}.building"

  if [[ ! -s "$onnx" ]]; then
    echo "ERROR: local ONNX model is missing: $onnx" >&2
    exit 1
  fi

  if [[ -s "$engine" && -s "$metadata" ]] && \
     [[ "$(cat "$metadata")" == "$fingerprint" ]] && \
     "$trtexec" --loadEngine="$engine" --skipInference >/dev/null 2>&1; then
    echo "ENGINE_OK $engine ($gpu_name, $trt_version)"
    return
  fi

  if [[ -e "$engine" ]]; then
    local rejected="${engine}.incompatible.$(date +%Y%m%d_%H%M%S)"
    mv "$engine" "$rejected"
    [[ ! -e "$metadata" ]] || mv "$metadata" "${rejected}.metadata"
    echo "ENGINE_REJECTED $rejected"
  fi

  rm -f "$temporary"
  echo "ENGINE_BUILD $engine from $onnx"
  "$trtexec" \
    --onnx="$onnx" \
    --minShapes=render_input:1x160x160x6,transf_input:1x160x160x6 \
    --optShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
    --maxShapes=render_input:252x160x160x6,transf_input:252x160x160x6 \
    --fp16 --memPoolSize=workspace:4096 --saveEngine="$temporary"
  "$trtexec" --loadEngine="$temporary" --skipInference >/dev/null
  mv "$temporary" "$engine"
  printf '%s\n' "$fingerprint" > "$metadata"
  echo "ENGINE_READY $engine"
}

ensure_engine refiner
ensure_engine scorer
