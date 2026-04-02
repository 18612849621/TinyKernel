# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

Each kernel lives in `kernels/<name>/` and consists of:
- `<name>.cu` — CUDA kernel + pybind11 bindings (compiled via `torch.utils.cpp_extension.load`)
- `<name>.py` — Python wrapper that JIT-compiles the `.cu` and exposes a clean API
- `benchmark.py` — correctness check vs `torch` + TFLOPS benchmark

GEMM implementations are grouped under `kernels/gemm/<variant>/` (e.g. `naive/`, `cutlass/`), each following the same three-file pattern with unified names `gemm.cu`, `gemm.py`, `benchmark.py`.

Shared CUDA utilities are in `kernels/cuda_utils.h`.

CUTLASS is in `thirdparty/cutlass/` as a git submodule (sparse checkout, `include/` only).

Standard NVCC flags (used in CMake and should be mirrored in JIT builds): `-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`.

## Running a kernel

```bash
cd kernels/<name>
python3 benchmark.py
```

On conda environments, `benchmark.py` must preload system libstdc++ before importing the extension:
```python
import ctypes, os
_lib = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_lib):
    ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
```

## Build (CMake alternative)

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
make -j$(nproc)
```

## Format

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files   # run manually
```

- Python: Black
- C++/CUDA: clang-format (Google style, 4-space indent, 100-col limit)

## Submodule setup

```bash
git submodule update --init --recursive
```
