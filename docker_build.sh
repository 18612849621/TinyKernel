#!/usr/bin/env bash
set -e

IMAGE="tinykernel:latest"

docker build -t "$IMAGE" "$(dirname "$0")"

echo ""
echo "Run with GPU:"
echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace $IMAGE bash"
