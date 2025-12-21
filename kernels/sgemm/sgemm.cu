#include <cuda_runtime.h>

#include <torch/extension.h>
#include <torch/types.h>

__device__ sgemm_native_f32_kernel(float* A, float* B, float* C, int M, int N, int K) {
	
}


void sgemm_native_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kFloat32)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kFloat32)
  const int M = a.size(0);
  const int N = a.size(1);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
	
}
