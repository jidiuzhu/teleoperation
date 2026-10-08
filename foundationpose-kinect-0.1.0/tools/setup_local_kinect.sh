#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_name="${FP_CONDA_ENV:-kinect}"
image="${FP_IMAGE:-foundationpose-cpp:trt10-ubuntu22.04}"
mode="${1:---all}"

usage() {
  cat <<'EOF'
Usage: ./tools/setup_local_kinect.sh [--all|--system|--conda|--image|--check]

  --all      Install host packages, the kinect Conda environment, and Docker image
  --system   Install Docker, NVIDIA Container Toolkit, Kinect SDK, and udev rules
  --conda    Create/update the kinect environment and install the local packages
  --image    Build the FoundationPose TensorRT Docker image
  --check    Print prerequisite and Python import status without installing

Environment overrides:
  FP_CONDA_ENV=name   Conda environment name (default: kinect)
  FP_IMAGE=name       Docker image name (default: foundationpose-cpp:trt10-ubuntu22.04)
EOF
}

case "$mode" in
  --all|--system|--conda|--image|--check) ;;
  -h|--help) usage; exit 0 ;;
  *) echo "ERROR: unknown option: $mode" >&2; usage >&2; exit 2 ;;
esac

if ((EUID == 0)); then
  echo "ERROR: run this script as your normal user; it invokes sudo only for host packages." >&2
  exit 1
fi

step() {
  printf '\n\033[1;36m==> %s\033[0m\n' "$1"
}

run_docker() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    sudo docker "$@"
  fi
}

install_system() {
  step "Install Ubuntu host packages"
  if [[ ! -r /etc/os-release ]]; then
    echo "ERROR: /etc/os-release is unavailable" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != ubuntu || ("${VERSION_ID:-}" != 22.04 && "${VERSION_ID:-}" != 24.04) ]]; then
    echo "ERROR: supported hosts are Ubuntu 22.04 and 24.04; found ${PRETTY_NAME:-unknown}" >&2
    exit 1
  fi

  sudo apt-get update
  sudo apt-get install -y --no-install-recommends \
    build-essential ca-certificates cmake curl gnupg lsb-release ninja-build \
    pkg-config unzip usbutils docker.io nvidia-container-toolkit \
    libgl1-mesa-dev libsoundio-dev libjpeg-dev libvulkan-dev libx11-dev \
    libxcursor-dev libxinerama-dev libxrandr-dev libusb-1.0-0-dev \
    libssl-dev libudev-dev mesa-common-dev uuid-dev

  step "Configure Docker GPU runtime"
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl enable --now docker
  sudo systemctl restart docker

  step "Install Azure Kinect SDK 1.4 and depth engine"
  if ! ldconfig -p 2>/dev/null | grep -q 'libk4a.so.1.4'; then
    sudo install -d -m 0755 /etc/apt/keyrings
    if [[ ! -s /etc/apt/keyrings/microsoft-prod.gpg ]]; then
      curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | sudo gpg --batch --yes --dearmor -o /etc/apt/keyrings/microsoft-prod.gpg
    fi
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/18.04/prod bionic main" \
      | sudo tee /etc/apt/sources.list.d/microsoft-k4a.list >/dev/null
    sudo apt-get update
    sudo env ACCEPT_EULA=Y apt-get install -y libk4a1.4 || true
    sudo ldconfig
    if ! ldconfig -p 2>/dev/null | grep -q 'libk4a.so.1.4'; then
      sudo "$root/tools/install_k4a_from_source.sh" \
        "$root/Azure-Kinect-Sensor-SDK-1.4.2.zip"
    fi
  else
    echo "Azure Kinect SDK is already registered."
  fi

  step "Install Kinect USB permissions"
  sudo tee /etc/udev/rules.d/99-azure-kinect.rules >/dev/null <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097a", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097b", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097c", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097d", GROUP="plugdev", MODE="0660", TAG+="uaccess"
EOF
  sudo udevadm control --reload-rules
  sudo udevadm trigger
  sudo usermod -aG docker,plugdev "$USER"
  echo "Host installation finished. Log out and back in after this script completes."
}

install_conda() {
  step "Create or update Conda environment: $env_name"
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in PATH" >&2
    exit 1
  fi

  conda_args=(--override-channels -c conda-forge -y
    python=3.11 'numpy>=1.26,<2' opencv=4.10 open3d=0.18
    paramiko=3.5 pillow pip=24.2 'setuptools<81')
  if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    conda install -n "$env_name" "${conda_args[@]}"
  else
    conda create -n "$env_name" "${conda_args[@]}"
  fi

  step "Install CUDA PyTorch with visible pip progress"
  conda run --no-capture-output -n "$env_name" \
    python -m pip install --progress-bar on \
      --extra-index-url https://download.pytorch.org/whl/cu126 \
      'torch==2.7.1+cu126' 'torchvision==0.22.1+cu126'

  step "Install Kinect and SAM3 Python dependencies"
  conda run --no-capture-output -n "$env_name" \
    python -m pip install --progress-bar on \
      pykinect-azure==0.0.4 einops pycocotools psutil \
      'timm>=1.0.17' tqdm 'ftfy==6.1.1' regex 'iopath>=0.1.10' \
      typing_extensions huggingface_hub
  conda run --no-capture-output -n "$env_name" \
    python -m pip install --progress-bar on --no-deps -e "$root/sam3"
  conda run --no-capture-output -n "$env_name" \
    python -m pip install --progress-bar on --no-deps -e "$root"
}

build_image() {
  step "Build FoundationPose TensorRT image: $image"
  if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: Docker is not installed; run $0 --system first" >&2
    exit 1
  fi
  # Ubuntu's docker.io package may use the legacy builder and omit buildx.
  # Plain docker build still streams every build step to the terminal.
  run_docker build \
    --file "$root/foundationpose_cpp/docker/Dockerfile.ubuntu22.04-trt10" \
    --tag "$image" "$root/foundationpose_cpp"

  step "Build TensorRT FP16 engines for this GPU"
  run_docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    -v "$root/foundationpose_cpp:/workspace" \
    -w /workspace "$image" \
    bash /workspace/tools/ensure_local_engines.sh
}

check_all() {
  step "Host checks"
  checks_failed=0
  check() {
    local label="$1"
    shift
    if "$@"; then
      echo "OK: $label"
    else
      echo "FAIL: $label" >&2
      checks_failed=$((checks_failed + 1))
    fi
  }
  check "NVIDIA driver" nvidia-smi -L
  check "Docker daemon" run_docker info
  if run_docker image inspect "$image" >/dev/null 2>&1; then
    check "Docker GPU access" run_docker run --rm --gpus all "$image" nvidia-smi -L
  else
    echo "FAIL: FoundationPose Docker image is missing: $image" >&2
    echo "Run: $root/tools/setup_local_kinect.sh --image" >&2
    checks_failed=$((checks_failed + 1))
  fi
  check "Azure Kinect SDK" bash -c "ldconfig -p | grep -q 'libk4a.so.1.4'"
  check "Azure Kinect depth engine" bash -c \
    "test -s /usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0 || \
     test -s /usr/local/lib/libdepthengine.so.2.0 || \
     ldconfig -p | grep -q 'libdepthengine.so.2.0'"
  check "Azure Kinect USB" bash -c \
    "lsusb | grep -Eiq '045e:(097a|097b|097c|097d)|Azure Kinect'"
  check "SAM3 checkpoint" test -s "$root/sam3/weights/sam3.pt"
  check "FoundationPose refiner ONNX" test -s "$root/foundationpose_cpp/models/refiner_hwc.onnx"
  check "FoundationPose scorer ONNX" test -s "$root/foundationpose_cpp/models/scorer_hwc.onnx"

  step "Conda imports"
  if conda env list | awk '{print $1}' | grep -qx "$env_name"; then
    conda run --no-capture-output -n "$env_name" python - <<'PY'
import cv2
import numpy
import open3d
import torch
from pykinect_azure import pykinect
import sam3
import foundationpose_kinect

print("numpy", numpy.__version__)
print("opencv", cv2.__version__)
print("open3d", open3d.__version__)
print("torch", torch.__version__)
print("torch.cuda.is_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("torch.cuda.device", torch.cuda.get_device_name(0))
print("Python imports: OK")
PY
  else
    echo "FAIL: Conda environment '$env_name' does not exist" >&2
    checks_failed=$((checks_failed + 1))
  fi

  if ((checks_failed)); then
    echo "$checks_failed host check(s) failed; fix them before running the live pipeline." >&2
    return 1
  fi
  echo "All installation checks passed."
}

case "$mode" in
  --system) install_system ;;
  --conda) install_conda ;;
  --image) build_image ;;
  --check) check_all ;;
  --all)
    install_system
    install_conda
    build_image
    check_all
    ;;
esac
