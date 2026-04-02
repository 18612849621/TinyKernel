import os

import torch
from torch.utils.cpp_extension import load

_dir = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_dir, "../../..")

lib = load(
    name="gemm_cutlass_lib",
    sources=[os.path.join(_dir, "gemm.cu")],
    extra_cuda_cflags=[
        "-O3",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "-std=c++17",
    ],
    extra_cflags=["-std=c++17"],
    extra_include_paths=[os.path.join(_root, "thirdparty/cutlass/include")],
    verbose=False,
)


def sgemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CUTLASS SGEMM: C = A @ B, inputs must be float32 CUDA tensors."""
    M, K = A.shape
    _, N = B.shape
    C = torch.zeros(M, N, dtype=torch.float32, device="cuda")
    lib.cutlass_sgemm(A, B, C)
    return C
