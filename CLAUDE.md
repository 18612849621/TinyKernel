# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Conventions

GEMM variants live in `kernels/gemm/<variant>/` with unified filenames: `gemm.cu`, `gemm.py`, `test.py`, `benchmark.py`.

Each `benchmark.py` covers perf-only shapes; `test.py` runs correctness against `A.cpu() @ B.cpu()`.

Standard NVCC flags (mirror in JIT builds): `-O3 --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math`.

Every `benchmark.py` / `test.py` must preload libstdc++ before importing any extension:
```python
import ctypes, os
_lib = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_lib):
    ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)
```
