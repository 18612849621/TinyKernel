import ctypes
import os
import time

_lib = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_lib):
    ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)

import torch
from torch.utils.cpp_extension import load

lib = load(
    name="hgemm_naive_lib",
    sources=[os.path.join(os.path.dirname(__file__), "gemm.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-std=c++17"],
    verbose=False,
)

torch.set_grad_enabled(False)

KERNELS = [
    ("naive", lib.hgemm_naive),
    ("tiled", lib.hgemm_tiled),
    ("thread_tiled", lib.hgemm_thread_tiled),
]

# fmt: off
SHAPES = [
    (4096, 4096,  4096),
    (8192, 8192,  8192),
    (4096, 16384, 4096),
    (8192, 8192,  128 ),
    (4096, 16384, 128 ),
]
# fmt: on


def bench(fn, A, B, iters=20, warmup=5):
    C = torch.zeros(A.size(0), B.size(1), dtype=torch.float16, device="cuda")
    for _ in range(warmup):
        fn(A, B, C)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn(A, B, C)
    torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / iters
    return ms, 2 * A.size(0) * B.size(1) * A.size(1) * 1e-12 / (ms * 1e-3)


for M, N, K in SHAPES:
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    print(f"{M}x{K} * {K}x{N} = {M}x{N}  (fp16)")
    for name, fn in KERNELS:
        ms, tflops = bench(fn, A, B)
        print(f"  {name:12s}  {ms:8.3f} ms  {tflops:.3f} TFLOPS")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        _ = A @ B
    torch.cuda.synchronize()
    ms_ref = (time.time() - t0) * 1000 / 20
    print(f"  {'cublas':12s}  {ms_ref:8.3f} ms  {2*M*N*K*1e-12/(ms_ref*1e-3):.3f} TFLOPS")
    print()
