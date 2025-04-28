#! /usr/bin/env bash

set -e

IMAGE_NAME="transformer_kernel_modeling"
TAG="latest"
PROJECT_NAME="transformer_kernel_modeling"
CONTAINER_BASE_NAME="${IMAGE_NAME}-container"
CONTAINER_PROJECT_PATH="${HOME}/${PROJECT_NAME}"


function usage() {
    echo "Usage: $0 {build|run|help}"
    echo "  build - Build and tag the docker image"
    echo "  run   - Get a shell in a container"
    echo "  help  - Print this message"
}

if [ $# -eq 0 ]
then
    usage
    exit
fi

case "${1}" in
    build)
        docker build \
               -t "${IMAGE_NAME}:${TAG}" \
               --build-arg USER_NAME="$(whoami)" \
               --build-arg PROJECT_NAME="${PROJECT_NAME}" \
               -f Dockerfile .
        ;;
    run)
        docker run --rm -it \
               --ipc=host \
               --privileged \
               --runtime=nvidia \
               --gpus all \
               -e NVIDIA_DRIVER_CAPABILITIES=all \
               --name "${CONTAINER_BASE_NAME}" \
               -v $(pwd):"${CONTAINER_PROJECT_PATH}" \
               "${IMAGE_NAME}" \
               /bin/bash
        ;;
    help|--help|-h|*)
        usage
        ;;
esac
