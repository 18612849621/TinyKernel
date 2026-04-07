import ctypes
import os

_lib = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_lib):
    ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)

import torch
from torch.utils.cpp_extension import load

lib = load(
    name="hgemm_naive_lib",
    sources=[os.path.join(os.path.dirname(__file__), "hgemm.cu")],
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
    (512,  512,  512 ),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (4096, 16384, 4096),
    (8192, 8192,  128 ),
    (4096, 16384, 128 ),
]
# fmt: on

for M, N, K in SHAPES:
    A = torch.randn(M, K, dtype=torch.float16, device="cuda")
    B = torch.randn(K, N, dtype=torch.float16, device="cuda")
    ref = A.cpu().to(torch.float32) @ B.cpu().to(torch.float32)
    print(f"{M}x{K} * {K}x{N} = {M}x{N}  (fp16)")
    for name, fn in KERNELS:
        C = torch.zeros(M, N, dtype=torch.float16, device="cuda")
        fn(A, B, C)
        err = (C.cpu().to(torch.float32) - ref).abs().max().item()
        status = "PASS" if err < 1e0 else "FAIL"
        print(f"  [{status}] {name:12s}  max_err={err:.6f}")
    print()
