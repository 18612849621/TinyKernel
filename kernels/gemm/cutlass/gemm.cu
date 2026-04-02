#include <torch/extension.h>

#include "cutlass/gemm/device/gemm.h"

// CUTLASS SGEMM: C = alpha * A * B + beta * C
// A: [M, K], B: [K, N], C: [M, N], all row-major float32
void cutlass_sgemm(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    TORCH_CHECK(A.dtype() == torch::kFloat32);
    TORCH_CHECK(B.dtype() == torch::kFloat32);
    TORCH_CHECK(C.dtype() == torch::kFloat32);
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda());

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(B.size(0) == K && C.size(0) == M && C.size(1) == N);

    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,   // A
        float, cutlass::layout::RowMajor,   // B
        float, cutlass::layout::RowMajor,   // C
        float                               // accumulator
    >;

    Gemm gemm_op;
    cutlass::Status status = gemm_op({
        {M, N, K},
        {A.data_ptr<float>(), K},
        {B.data_ptr<float>(), N},
        {C.data_ptr<float>(), N},
        {C.data_ptr<float>(), N},
        {1.0f, 0.0f}
    });
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_sgemm", &cutlass_sgemm, "CUTLASS SGEMM (CUDA)");
}
