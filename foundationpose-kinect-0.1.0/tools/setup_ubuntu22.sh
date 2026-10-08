#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mode="${1:---check}"
if [[ "$mode" != "--check" && "$mode" != "--install" ]]; then
  echo "Usage: $0 [--check|--install]" >&2
  exit 2
fi

if [[ ! -r /etc/os-release ]]; then
  echo "ERROR: /etc/os-release is unavailable" >&2
  exit 1
fi
# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "22.04" ]]; then
  echo "ERROR: this installer supports Ubuntu 22.04 only (found ${PRETTY_NAME:-unknown})" >&2
  exit 1
fi

install_packages() {
  if [[ $EUID -ne 0 ]]; then
    echo "ERROR: run the install mode with sudo" >&2
    exit 1
  fi

  install -d -m 0755 /etc/apt/keyrings
  apt-get update
  apt-get install -y ca-certificates curl gnupg lsb-release ubuntu-drivers-common usbutils

  if ! nvidia-smi -L >/dev/null 2>&1; then
    ubuntu-drivers install
    echo "NVIDIA driver installed or updated; reboot before the final --check."
  fi

  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list

  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --batch --yes --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

  curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
    | gpg --batch --yes --dearmor -o /etc/apt/keyrings/microsoft-prod.gpg
  echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/ubuntu/18.04/prod bionic main" \
    > /etc/apt/sources.list.d/microsoft-k4a.list

  apt-get update
  apt-get install -y \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl enable --now docker

  # Microsoft's old Ubuntu 18.04 development packages are not consistently
  # installable on Jammy. Prefer the runtime package, then build the open SDK.
  ACCEPT_EULA=Y apt-get install -y libk4a1.4 || true
  ldconfig
  if ! ldconfig -p | grep -q 'libk4a.so.1.4'; then
    "$root/tools/install_k4a_from_source.sh"
  fi

  cat > /etc/udev/rules.d/99-azure-kinect.rules <<'EOF'
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097a", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097b", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097c", GROUP="plugdev", MODE="0660", TAG+="uaccess"
SUBSYSTEM=="usb", ATTR{idVendor}=="045e", ATTR{idProduct}=="097d", GROUP="plugdev", MODE="0660", TAG+="uaccess"
EOF
  udevadm control --reload-rules
  udevadm trigger

  local login_user="${SUDO_USER:-}"
  if [[ -n "$login_user" ]]; then
    usermod -aG docker,plugdev "$login_user"
    echo "Added $login_user to docker and plugdev; log out and back in once."
  fi
}

if [[ "$mode" == "--install" ]]; then
  install_packages
fi

failures=0
check() {
  local label="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "OK: $label"
  else
    echo "FAIL: $label" >&2
    failures=$((failures + 1))
  fi
}

check "NVIDIA driver sees a GPU" nvidia-smi -L
check "Docker daemon is reachable" docker info
check "NVIDIA Container Toolkit exposes the GPU" \
  docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
check "Azure Kinect SDK 1.4 library is registered" \
  bash -c "ldconfig -p | grep -q 'libk4a.so.1.4'"
check "Azure Kinect USB device is present" \
  bash -c "lsusb | grep -Eiq '045e:(097a|097b|097c|097d)|Azure Kinect'"

if ((failures > 0)); then
  echo "$failures prerequisite check(s) failed." >&2
  exit 1
fi
echo "Ubuntu host prerequisites are ready."
