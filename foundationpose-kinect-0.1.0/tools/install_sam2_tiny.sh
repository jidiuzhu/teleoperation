#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${SAM2_CONDA_ENV:-kinect}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_DIR="${PROJECT_ROOT}/weights"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/sam2.1_hiera_tiny.pt"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is not available in PATH" >&2
  exit 1
}
command -v curl >/dev/null 2>&1 || {
  echo "ERROR: curl is not installed" >&2
  exit 1
}

echo "==> Install SAM2 runtime into conda environment: ${ENV_NAME}"
conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install hydra-core==1.3.2 'iopath>=0.1.10' omegaconf
SAM2_BUILD_CUDA=0 conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install --no-deps --no-build-isolation \
  'git+https://github.com/facebookresearch/sam2.git'

mkdir -p "${CHECKPOINT_DIR}"
CHECKPOINT_MIN_BYTES=100000000
EXISTING_BYTES=0
if [[ -f "${CHECKPOINT_PATH}" ]]; then
  EXISTING_BYTES="$(stat -c %s -- "${CHECKPOINT_PATH}")"
fi
if (( EXISTING_BYTES >= CHECKPOINT_MIN_BYTES )); then
  echo "==> Checkpoint already exists: ${CHECKPOINT_PATH}"
else
  echo "==> Download SAM2.1 Hiera Tiny checkpoint (~149 MB)"
  TEMP_CHECKPOINT="${CHECKPOINT_PATH}.download"
  trap 'rm -f -- "${TEMP_CHECKPOINT}"' EXIT
  curl --fail --location --progress-bar \
    --output "${TEMP_CHECKPOINT}" "${CHECKPOINT_URL}"
  DOWNLOADED_BYTES="$(stat -c %s -- "${TEMP_CHECKPOINT}")"
  if (( DOWNLOADED_BYTES < CHECKPOINT_MIN_BYTES )); then
    echo "ERROR: downloaded checkpoint is unexpectedly small (${DOWNLOADED_BYTES} bytes)" >&2
    exit 1
  fi
  mv -- "${TEMP_CHECKPOINT}" "${CHECKPOINT_PATH}"
  trap - EXIT
fi

echo "==> Verify SAM2 imports and checkpoint"
SAM2_CHECKPOINT="${CHECKPOINT_PATH}" conda run --no-capture-output -n "${ENV_NAME}" \
  python -c 'import os; from pathlib import Path; from sam2.build_sam import build_sam2; from sam2.sam2_image_predictor import SAM2ImagePredictor; p=Path(os.environ["SAM2_CHECKPOINT"]); print("SAM2 import: OK"); print(f"Checkpoint: {p} ({p.stat().st_size / 1024**2:.1f} MiB)")'

echo "==> SAM2 Tiny installation complete"
echo "Run Cutie tracking with SAM2 box refinement from this project root with:"
echo "python teleop/scripts/kinect_foundationpose_publisher.py --sam2 --fps 15 --viewer"
