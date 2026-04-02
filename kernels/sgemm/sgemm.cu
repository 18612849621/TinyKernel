#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#define TILE 32

// Naive SGEMM: each thread computes one C[row, col]
__global__ void sgemm_naive_f32_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// Tiled shared-memory SGEMM
__global__ void sgemm_tiled_f32_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int kA = t * TILE + threadIdx.x;
        int kB = t * TILE + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? A[row * K + kA] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (kB < K && col < N) ? B[kB * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

static void launch(
    void (*kernel)(const float*, const float*, float*, int, int, int),
    torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

void sgemm_naive_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    launch(sgemm_naive_f32_kernel, A, B, C);
}

void sgemm_tiled_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    launch(sgemm_tiled_f32_kernel, A, B, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_naive_f32", &sgemm_naive_f32, "Naive SGEMM FP32");
    m.def("sgemm_tiled_f32", &sgemm_tiled_f32, "Tiled shared-mem SGEMM FP32");
}
