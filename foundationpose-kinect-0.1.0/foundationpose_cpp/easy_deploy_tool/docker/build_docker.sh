#!/bin/bash

IMAGE_BASE_NAME="easy_deploy_base_dev"
BUILT_IMAGE_TAG=""
EXTERNAL_TAG=""

script_dir="$( cd "$(dirname "$0")" && pwd )"
parent_dir="$( cd "$script_dir/../.." && pwd )"
parent_dir_name="$(basename "$parent_dir")"

CONTAINER_NAME="easy_deploy_${parent_dir_name}"

BUILD_FUNCTIONS=(
  nvidia_gpu_trt8_u2004
  nvidia_gpu_trt8_u2204
  nvidia_gpu_trt10_u2204
  jetson_trt8_u2004
  jetson_trt8_u2204
  jetson_trt10_u2204
  rknn_230_u2204
)

is_image_exist() {
  local name="$1"
  if docker images --filter "reference=$name" \
                   --format "{{.Repository}}:{{.Tag}}" | grep -q "$name"; then
    return 0
  else
    return 1
  fi
}

is_container_exist() {
  local name="$1"
  if docker ps -a --filter "name=$name" | grep -q "$name"; then
    return 0
  else
    return 1
  fi
}

build_image() {
  local image_full_name="${IMAGE_BASE_NAME}:${BUILT_IMAGE_TAG}"
  if is_image_exist ${image_full_name}; then
    echo Image: ${image_full_name} exists! Skip image building process ...
    return 1
  else
    docker build -f "${script_dir}/${DOCKER_FILE_NAME}" -t "${image_full_name}" .
    return 0
  fi
}

create_container() {
  local image_full_name="${IMAGE_BASE_NAME}:${BUILT_IMAGE_TAG}"
  if ! is_image_exist ${image_full_name}; then
    echo Image: ${image_full_name} does not exist, quit creating ...
    return 1
  fi

  if is_container_exist ${CONTAINER_NAME}; then
    echo Container: ${CONTAINER_NAME} exists! Skip container building process ...
    return 0
  fi

  docker run -itd --privileged \
             --device /dev/dri \
             --group-add video \
             -v /tmp/.X11-unix:/tmp/.X11-unix \
             --network host \
             --ipc host \
             -v ${parent_dir}:/workspace \
             -w /workspace \
             -v /dev/bus/usb:/dev/bus/usb \
             -e DISPLAY=${DISPLAY} \
             -e DOCKER_USER=${USER} \
             -e USER=${USER} \
             --name ${CONTAINER_NAME} \
             ${EXTERNAL_TAG} \
             ${image_full_name} \
             /bin/bash

  return 0
}

wrap_function() {
    local func_name=$1

    if ! declare -f "$func_name" >/dev/null; then
        echo "错误：函数 $func_name 未定义"
        return 1
    fi

    local original_func=$(declare -f "$func_name")

    eval "
        $func_name() {
            $func_name-original \"\$@\"  # 原始函数
            build_image \"\$@\"
            create_container \"\$@\"
        }
    "

    eval "${original_func/$func_name/$func_name-original}"
}


nvidia_gpu_trt8_u2004() {
  BUILT_IMAGE_TAG=nvidia_gpu_tensorrt_trt8_u2004
  DOCKER_FILE_NAME="nvidia_gpu_tensorrt_trt8_u2004.dockerfile"
  EXTERNAL_TAG="--runtime nvidia"
}

nvidia_gpu_trt8_u2204() {
  BUILT_IMAGE_TAG=nvidia_gpu_tensorrt_trt8_u2204
  DOCKER_FILE_NAME="nvidia_gpu_tensorrt_trt8_u2204.dockerfile"
  EXTERNAL_TAG="--runtime nvidia"
}

nvidia_gpu_trt10_u2204() {
  BUILT_IMAGE_TAG=nvidia_gpu_tensorrt_trt10_u2204
  DOCKER_FILE_NAME="nvidia_gpu_tensorrt_trt10_u2204.dockerfile"
  EXTERNAL_TAG="--runtime nvidia"
}

jetson_trt8_u2004() {
  BUILT_IMAGE_TAG=jetson_tensorrt_trt8_u2004
  DOCKER_FILE_NAME="jetson_tensorrt_trt8_u2004.dockerfile"
  EXTERNAL_TAG="--runtime nvidia"
}

jetson_trt8_u2204() {
  BUILT_IMAGE_TAG=jetson_tensorrt_trt8_u2204
  DOCKER_FILE_NAME="jetson_tensorrt_trt8_u2204.dockerfile"
  EXTERNAL_TAG="--runtime nvidia"
}

jetson_trt10_u2204() {
  BUILT_IMAGE_TAG=jetson_tensorrt_trt10_u2204
  DOCKER_FILE_NAME="jetson_tensorrt_trt10_u2204.dockerfile"
  EXTERNAL_TAG="--runtime nvidia"
}

rknn_230_u2204() {
  BUILT_IMAGE_TAG=rknn_230_u2204
  DOCKER_FILE_NAME="rknn_230_u2204.dockerfile"
  EXTERNAL_TAG=""
}

register_with_base() {
    for fname in "${BUILD_FUNCTIONS[@]}"; do
        wrap_function "$fname"
    done
}
register_with_base

if [ $# -gt 0 ]; then
  # 执行指定的函数
  "$@"
fi
