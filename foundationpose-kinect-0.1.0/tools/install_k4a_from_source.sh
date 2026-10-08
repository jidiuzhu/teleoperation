#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_input="${1:-}"
if ((EUID != 0)); then
  echo "ERROR: run with sudo: sudo $0 [SDK_SOURCE_DIR_OR_ZIP]" >&2
  exit 1
fi
if [[ -z "$source_input" ]]; then
  if [[ -d "$root/Azure-Kinect-Sensor-SDK-1.4.2" ]]; then
    source_input="$root/Azure-Kinect-Sensor-SDK-1.4.2"
  elif [[ -f "$root/Azure-Kinect-Sensor-SDK-1.4.2.zip" ]]; then
    source_input="$root/Azure-Kinect-Sensor-SDK-1.4.2.zip"
  else
    echo "ERROR: Azure Kinect SDK 1.4.2 source directory or zip was not found" >&2
    exit 1
  fi
fi
[[ "$source_input" == /* ]] || source_input="$PWD/$source_input"

apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build pkg-config unzip nasm \
  libgl1-mesa-dev libsoundio-dev libjpeg-dev libvulkan-dev libx11-dev \
  libxcursor-dev libxinerama-dev libxrandr-dev libusb-1.0-0-dev \
  libssl-dev libudev-dev mesa-common-dev uuid-dev

temporary="$(mktemp -d /tmp/k4a-source-install.XXXXXX)"
cleanup() { rm -rf "$temporary"; }
trap cleanup EXIT
if [[ -d "$source_input" ]]; then
  source_directory="$source_input"
elif [[ -f "$source_input" ]]; then
  unzip -q "$source_input" -d "$temporary/source"
  source_directory="$(find "$temporary/source" -mindepth 1 -maxdepth 1 -type d | head -1)"
else
  echo "ERROR: SDK source does not exist: $source_input" >&2
  exit 1
fi

cmake -S "$source_directory" -B "$temporary/build" -GNinja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DK4A_BUILD_DOCS=OFF -DK4A_VALIDATE_CLANG_FORMAT=OFF
cmake --build "$temporary/build" --parallel "$(nproc)"
cmake --install "$temporary/build"

depth_engine="${K4A_DEPTHENGINE_PATH:-}"
if [[ -z "$depth_engine" ]]; then
  depth_engine="$(find "$source_directory" /usr/lib /opt -type f \
    -name 'libdepthengine.so.2.0' -print -quit 2>/dev/null || true)"
fi
if [[ -z "$depth_engine" ]]; then
  echo "Azure Kinect depth engine was not found; trying the Microsoft runtime package."
  ACCEPT_EULA=Y apt-get install -y libk4a1.4 || true
  depth_engine="$(find /usr/lib /opt -type f -name 'libdepthengine.so.2.0' \
    -print -quit 2>/dev/null || true)"
fi
if [[ -z "$depth_engine" ]]; then
  # Downloading and extracting the runtime deb does not require its obsolete
  # Ubuntu 18.04 dependencies to be installable on Jammy.
  (cd "$temporary" && apt-get download libk4a1.4) || true
  runtime_deb="$(find "$temporary" -maxdepth 1 -type f -name 'libk4a1.4*.deb' \
    -print -quit 2>/dev/null || true)"
  if [[ -n "$runtime_deb" ]]; then
    dpkg-deb -x "$runtime_deb" "$temporary/runtime"
    depth_engine="$(find "$temporary/runtime" -type f -name 'libdepthengine.so.2.0' \
      -print -quit 2>/dev/null || true)"
  fi
fi
if [[ -z "$depth_engine" ]]; then
  echo "ERROR: libdepthengine.so.2.0 is required but is not part of the open-source SDK." >&2
  echo "Install Microsoft's libk4a1.4 runtime or rerun with:" >&2
  echo "  sudo K4A_DEPTHENGINE_PATH=/path/libdepthengine.so.2.0 $0 $source_input" >&2
  exit 1
fi
install -m 0644 "$depth_engine" /usr/local/lib/libdepthengine.so.2.0
install -m 0644 "$source_directory/scripts/99-k4a.rules" /etc/udev/rules.d/99-k4a.rules
ldconfig
udevadm control --reload-rules
udevadm trigger
echo "Azure Kinect SDK source installation completed."
