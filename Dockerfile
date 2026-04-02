FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ninja-build git \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu126

WORKDIR /workspace
