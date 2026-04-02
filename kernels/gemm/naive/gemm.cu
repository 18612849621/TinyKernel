#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

#define TILE 32
#define THREAD_TILE 4

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

// Thread-tiled SGEMM: each thread computes THREAD_TILE x THREAD_TILE elements
__global__ void sgemm_thread_tiled_f32_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int block_row = blockIdx.y * TILE;
    int block_col = blockIdx.x * TILE;
    int thread_row = threadIdx.y * THREAD_TILE;
    int thread_col = threadIdx.x * THREAD_TILE;

    float acc[THREAD_TILE][THREAD_TILE] = {};

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        // load tile into shared memory
        for (int tr = 0; tr < THREAD_TILE; ++tr) {
            for (int tc = 0; tc < THREAD_TILE; ++tc) {
                int global_row = block_row + thread_row + tr;
                int global_k = t * TILE + thread_col + tc;
                sA[thread_row + tr][thread_col + tc] =
                    (global_row < M && global_k < K) ? A[global_row * K + global_k] : 0.f;

                int global_k2 = t * TILE + thread_row + tr;
                int global_col = block_col + thread_col + tc;
                sB[thread_row + tr][thread_col + tc] =
                    (global_k2 < K && global_col < N) ? B[global_k2 * N + global_col] : 0.f;
            }
        }
        __syncthreads();

        // compute on shared memory
        for (int k = 0; k < TILE; ++k) {
            for (int tr = 0; tr < THREAD_TILE; ++tr) {
                for (int tc = 0; tc < THREAD_TILE; ++tc) {
                    acc[tr][tc] += sA[thread_row + tr][k] * sB[k][thread_col + tc];
                }
            }
        }
        __syncthreads();
    }

    // write back
    for (int tr = 0; tr < THREAD_TILE; ++tr) {
        for (int tc = 0; tc < THREAD_TILE; ++tc) {
            int global_row = block_row + thread_row + tr;
            int global_col = block_col + thread_col + tc;
            if (global_row < M && global_col < N)
                C[global_row * N + global_col] = acc[tr][tc];
        }
    }
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

void sgemm_thread_tiled_f32(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(TILE / THREAD_TILE, TILE / THREAD_TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    sgemm_thread_tiled_f32_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_naive_f32", &sgemm_naive_f32, "Naive SGEMM FP32");
    m.def("sgemm_tiled_f32", &sgemm_tiled_f32, "Tiled shared-mem SGEMM FP32");
    m.def("sgemm_thread_tiled_f32", &sgemm_thread_tiled_f32, "Thread-tiled SGEMM FP32");
}
