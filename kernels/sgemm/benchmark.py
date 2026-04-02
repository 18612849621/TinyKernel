import ctypes
import os
import time

_lib = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_lib):
    ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)

import torch
from torch.utils.cpp_extension import load

lib = load(
    name="sgemm_lib",
    sources=[os.path.join(os.path.dirname(__file__), "sgemm.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-std=c++17"],
    verbose=False,
)

torch.set_grad_enabled(False)


def run(fn, A, B, C, iters=20, warmup=3):
    for _ in range(warmup):
        C.zero_()
        fn(A, B, C)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        C.zero_()
        fn(A, B, C)
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / iters
    M, K, N = A.size(0), A.size(1), B.size(1)
    tflops = 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return ms, tflops


# ── correctness ──────────────────────────────────────────────────────────────
M, K, N = 512, 512, 512
A = torch.randn(M, K, dtype=torch.float32, device="cuda")
B = torch.randn(K, N, dtype=torch.float32, device="cuda")
C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
ref = A @ B

for name, fn in [("naive", lib.sgemm_naive_f32), ("tiled", lib.sgemm_tiled_f32)]:
    C.zero_()
    fn(A, B, C)
    err = (C - ref).abs().max().item()
    status = "PASS" if err < 1e-2 else "FAIL"
    print(f"[{status}] {name:6s}  max_err={err:.6f}")

# ── benchmark ─────────────────────────────────────────────────────────────────
print()
for M, K, N in [(1024, 1024, 1024), (4096, 4096, 4096)]:
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    print(f"M={M} K={K} N={N}")
    for name, fn in [("naive", lib.sgemm_naive_f32), ("tiled", lib.sgemm_tiled_f32)]:
        ms, tflops = run(fn, A, B, C)
        print(f"  {name:6s}  {ms:7.3f} ms  {tflops:.2f} TFLOPS")
    # torch cublas reference
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        _ = A @ B
    torch.cuda.synchronize()
    ms_ref = (time.time() - t0) * 1000 / 20
    tflops_ref = 2 * M * N * K * 1e-12 / (ms_ref * 1e-3)
    print(f"  {'cublas':6s}  {ms_ref:7.3f} ms  {tflops_ref:.2f} TFLOPS")
