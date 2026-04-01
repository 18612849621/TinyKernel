# TinyKernel

High-performance CUDA kernels with PyTorch bindings.

## Structure

```
TinyKernel/
├── kernels/
│   ├── cuda_utils.h              # Shared CUDA utilities
│   ├── sgemm/                    # Naive SGEMM baseline
│   └── cutlass_gemm/             # CUTLASS SGEMM
│       ├── cutlass_gemm.cu       # CUTLASS kernel + pybind11
│       ├── cutlass_gemm.py       # Python wrapper (JIT build)
│       └── benchmark.py          # Correctness check + benchmark
├── thirdparty/
│   └── cutlass/                  # NVIDIA CUTLASS (submodule)
├── CMakeLists.txt
└── README.md
```

## Requirements

- CUDA 12+
- PyTorch 2.0+ (with CUDA)
- CMake 3.18+

## Quick Start

### 1. Clone with submodules

```bash
git clone --recurse-submodules https://github.com/<your-org>/TinyKernel.git
# or if already cloned:
git submodule update --init --recursive
```

### 2. Run CUTLASS GEMM (JIT via torch.utils.cpp_extension)

```bash
cd kernels/cutlass_gemm
python benchmark.py
```

Expected output:
```
Max error vs torch: 0.001007
M=4096 N=4096 K=4096: 2.645 ms, 51.96 TFLOPS
```

> **Note (conda users):** On systems where conda's `libstdc++` is older than the system's,
> `benchmark.py` preloads `/usr/lib/x86_64-linux-gnu/libstdc++.so.6` via `ctypes` before
> importing the extension. This is handled automatically.

### 3. Build with CMake (optional)

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
make -j$(nproc)
```

## Kernels

| Kernel | File | Description |
|--------|------|-------------|
| CUTLASS SGEMM | `kernels/cutlass_gemm/` | FP32 GEMM via CUTLASS device API |
| Naive SGEMM | `kernels/sgemm/` | Baseline FP32 GEMM |

## Format

```bash
pip install pre-commit
pre-commit install
```
