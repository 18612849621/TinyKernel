import ctypes
import os

_lib = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_lib):
    ctypes.CDLL(_lib, mode=ctypes.RTLD_GLOBAL)

import torch
from torch.utils.cpp_extension import load

lib = load(
    name="gemm_naive_lib",
    sources=[os.path.join(os.path.dirname(__file__), "gemm.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math"],
    extra_cflags=["-std=c++17"],
    verbose=False,
)

torch.set_grad_enabled(False)

KERNELS = [("naive", lib.sgemm_naive_f32), ("tiled", lib.sgemm_tiled_f32)]

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
    A = torch.randn(M, K, dtype=torch.float32, device="cuda")
    B = torch.randn(K, N, dtype=torch.float32, device="cuda")
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    ref = A.cpu() @ B.cpu()
    print(f"{M}x{K} * {K}x{N} = {M}x{N}  (fp32)")
    for name, fn in KERNELS:
        C.zero_()
        fn(A, B, C)
        err = (C.cpu() - ref).abs().max().item()
        status = "PASS" if err < 1e-2 else "FAIL"
        print(f"  [{status}] {name:6s}  max_err={err:.6f}")
    print()
