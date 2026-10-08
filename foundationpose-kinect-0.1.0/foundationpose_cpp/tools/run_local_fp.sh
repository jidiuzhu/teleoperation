#!/usr/bin/env bash
set -euo pipefail

cd /workspace

/workspace/tools/ensure_local_engines.sh

cmake -S . -B build -DENABLE_TENSORRT=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel "$(nproc)" --target foundationpose_local
./build/bin/foundationpose_local --dataset /workspace/test_data/apple \
  --output /workspace/test_output/apple --limit "${FP_LIMIT:-10}"
