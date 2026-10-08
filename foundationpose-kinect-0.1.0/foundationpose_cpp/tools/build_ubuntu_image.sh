#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${FP_IMAGE:-foundationpose-cpp:trt10-ubuntu22.04}"
proxy_args=()
network_args=()
for variable in HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY; do
  lowercase="${variable,,}"
  value="${!variable:-${!lowercase:-}}"
  if [[ -n "$value" ]]; then
    proxy_args+=(--build-arg "$variable=$value")
    proxy_args+=(--build-arg "$lowercase=$value")
  fi
done
if ((${#proxy_args[@]} > 0)); then
  network_args+=(--network=host)
fi

exec docker build \
  "${proxy_args[@]}" \
  "${network_args[@]}" \
  --file "$root/docker/Dockerfile.ubuntu22.04-trt10" \
  --tag "$image" \
  "$root"
