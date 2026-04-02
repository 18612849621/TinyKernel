import ctypes
import os
import time

_libstdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_libstdcpp):
    ctypes.CDLL(_libstdcpp, mode=ctypes.RTLD_GLOBAL)

import torch
from gemm import sgemm

torch.set_grad_enabled(False)

# fmt: off
# (M, N, K, label)
SHAPES = [
    # correctness (small)
    (512,  512,  512,  "correctness"),
    # large MNK — CUTLASS profiler / LLaMA-2 FFN shapes
    (4096, 4096,  4096, "large-mnk"),
    (8192, 8192,  8192, "large-mnk"),
    (4096, 16384, 4096, "large-mnk"),
    # large MN, small K — LoRA / MoE gating
    (8192, 8192,  128,  "large-mn-small-k"),
    (4096, 16384, 128,  "large-mn-small-k"),
]
# fmt: on


def bench(fn, A, B, iters=20, warmup=5):
    for _ in range(warmup):
        fn(A, B)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(A, B)
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / iters
    M, K, N = A.size(0), A.size(1), B.size(0)
    return ms, 2 * M * N * K * 1e-12 / (ms * 1e-3)


for M, N, K, label in SHAPES:
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    ref = A @ B
    print(f"[{label}] {M}x{K} * {K}x{N} = {M}x{N}  (fp32)")

    C = sgemm(A, B)
    err = (C - ref).abs().max().item()
    if label == "correctness":
        status = "PASS" if err < 1.0 else "FAIL"
        print(f"  [{status}] cutlass  max_err={err:.6f}")
    else:
        ms, tflops = bench(sgemm, A, B)
        print(f"  {'cutlass':8s}  {ms:8.3f} ms  {tflops:.3f} TFLOPS")

        # cublas reference
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            _ = A @ B
        torch.cuda.synchronize()
        ms_ref = (time.time() - t0) * 1000 / 20
        tflops_ref = 2 * M * N * K * 1e-12 / (ms_ref * 1e-3)
        print(f"  {'cublas':8s}  {ms_ref:8.3f} ms  {tflops_ref:.3f} TFLOPS")
    print()
