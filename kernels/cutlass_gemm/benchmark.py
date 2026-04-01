import ctypes
import os

# Must preload system libstdc++ before any extension .so is imported
# to avoid GLIBCXX version mismatch between conda and system gcc
_libstdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_libstdcpp):
    ctypes.CDLL(_libstdcpp, mode=ctypes.RTLD_GLOBAL)

import torch
from cutlass_gemm import sgemm

torch.set_grad_enabled(False)

M, K, N = 4096, 4096, 4096
A = torch.randn(M, K, dtype=torch.float32, device="cuda")
B = torch.randn(K, N, dtype=torch.float32, device="cuda")

# Correctness check
C_cutlass = sgemm(A, B)
C_ref = A @ B
max_err = (C_cutlass - C_ref).abs().max().item()
print(f"Max error vs torch: {max_err:.6f}")
assert max_err < 1.0, "Correctness check failed"

# Benchmark
import time

warmup, iters = 5, 20
for _ in range(warmup):
    sgemm(A, B)
torch.cuda.synchronize()

start = time.time()
for _ in range(iters):
    sgemm(A, B)
torch.cuda.synchronize()
elapsed_ms = (time.time() - start) * 1000 / iters

tflops = 2 * M * N * K * 1e-12 / (elapsed_ms * 1e-3)
print(f"M={M} N={N} K={K}: {elapsed_ms:.3f} ms, {tflops:.2f} TFLOPS")
