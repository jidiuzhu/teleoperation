#!/bin/bash

script_dir="$( cd "$(dirname "$0")" && pwd )"
parent_dir="$( cd "$script_dir/../.." && pwd )"
parent_dir_name="$(basename "$parent_dir")"

CONTAINER_NAME="easy_deploy_${parent_dir_name}"

docker start $CONTAINER_NAME
docker exec -it $CONTAINER_NAME /bin/bash
