#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_env="${FP_CONDA_ENV:-kinect}"
create_envs=0
build_image=0
install_package=0
while (($#)); do
  case "$1" in
    --envs) create_envs=1; shift ;;
    --image) build_image=1; shift ;;
    --python-package) install_package=1; shift ;;
    --all) create_envs=1; build_image=1; install_package=1; shift ;;
    -h|--help)
      echo "Usage: $0 [--envs] [--image] [--python-package] [--all]"
      exit 0 ;;
    *) echo "ERROR: unknown option: $1" >&2; exit 2 ;;
  esac
done

if ((create_envs)); then
  if conda env list | awk '{print $1}' | grep -qx "$conda_env"; then
    echo "Conda environment already exists: $conda_env"
  else
    conda env create -f "$root/environments/kinect.yml" -n "$conda_env"
  fi
fi
if ((install_package)); then
  conda run --no-capture-output -n "$conda_env" \
    python -m pip install --no-deps -e "$root"
fi
if ((build_image)); then
  "$root/foundationpose_cpp/tools/build_ubuntu_image.sh"
fi

echo "Module installation tasks completed. Run:"
echo "  conda run -n $conda_env fp-kinect inspect-config --root $root"
