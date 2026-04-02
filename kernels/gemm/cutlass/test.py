import ctypes
import os

_libstdcpp = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
if os.path.exists(_libstdcpp):
    ctypes.CDLL(_libstdcpp, mode=ctypes.RTLD_GLOBAL)

import torch
from gemm import sgemm

torch.set_grad_enabled(False)

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
    ref = A.cpu() @ B.cpu()
    C = sgemm(A, B)
    err = (C.cpu() - ref).abs().max().item()
    status = "PASS" if err < 1e-2 else "FAIL"
    print(f"{M}x{K} * {K}x{N} = {M}x{N}  (fp32)")
    print(f"  [{status}] cutlass  max_err={err:.6f}")
    print()
