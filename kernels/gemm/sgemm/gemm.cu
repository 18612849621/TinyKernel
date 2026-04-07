#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/types.h>

// ── configuration ────────────────────────────────────────────────────────────
// Block tile: each block computes BM x BN output, loading BK elements along K
constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;
// Thread tile: each thread computes TM x TN elements
constexpr int TM = 4;
constexpr int TN = 4;
// ─────────────────────────────────────────────────────────────────────────────

// Naive SGEMM: no tiling, each thread computes one C[row, col]
__global__ void sgemm_naive_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// Tiled shared-memory SGEMM: block tile BM x BN, K dimension tiled by BK
template <int _BM, int _BN, int _BK>
__global__ void sgemm_tiled_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
    __shared__ float sA[_BM][_BK];
    __shared__ float sB[_BK][_BN];

    int row = blockIdx.y * _BM + threadIdx.y;
    int col = blockIdx.x * _BN + threadIdx.x;
    float sum = 0.f;

    for (int t = 0; t < (K + _BK - 1) / _BK; ++t) {
        int kA = t * _BK + threadIdx.x;
        int kB = t * _BK + threadIdx.y;
        sA[threadIdx.y][threadIdx.x] = (row < M && kA < K) ? A[row * K + kA] : 0.f;
        sB[threadIdx.y][threadIdx.x] = (kB < K && col < N) ? B[kB * N + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < _BK; ++k) sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// Thread-tiled SGEMM: each thread computes TM x TN elements
// k is inner loop; sA value is loaded once per row into register and reused
// across all TN columns
template <int _BM, int _BN, int _BK, int _TM, int _TN>
__global__ void sgemm_thread_tiled_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    int M, int N, int K) {
    __shared__ float sA[_BM][_BK];
    __shared__ float sB[_BK][_BN];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int block_row = blockIdx.y * _BM;
    int block_col = blockIdx.x * _BN;
    int thread_row = ty * _TM;
    int thread_col = tx * _TN;

    float acc[_TM][_TN] = {};

    for (int t = 0; t < (K + _BK - 1) / _BK; ++t) {
        for (int tr = 0; tr < _TM; ++tr) {
            for (int tc = 0; tc < _TN; ++tc) {
                int global_row = block_row + thread_row + tr;
                int global_k = t * _BK + thread_col + tc;
                sA[thread_row + tr][thread_col + tc] =
                    (global_row < M && global_k < K) ? A[global_row * K + global_k] : 0.f;

                int global_k2 = t * _BK + thread_row + tr;
                int global_col = block_col + thread_col + tc;
                sB[thread_row + tr][thread_col + tc] =
                    (global_k2 < K && global_col < N) ? B[global_k2 * N + global_col] : 0.f;
            }
        }
        __syncthreads();

        for (int k = 0; k < _BK; ++k) {
            float a_reg[_TM];
            for (int tr = 0; tr < _TM; ++tr)
                a_reg[tr] = sA[thread_row + tr][k];
            for (int tc = 0; tc < _TN; ++tc)
                for (int tr = 0; tr < _TM; ++tr)
                    acc[tr][tc] += a_reg[tr] * sB[k][thread_col + tc];
        }
        __syncthreads();
    }

    for (int tr = 0; tr < _TM; ++tr) {
        for (int tc = 0; tc < _TN; ++tc) {
            int global_row = block_row + thread_row + tr;
            int global_col = block_col + thread_col + tc;
            if (global_row < M && global_col < N)
                C[global_row * N + global_col] = acc[tr][tc];
        }
    }
}

// ── launch helpers ────────────────────────────────────────────────────────────

void sgemm_naive(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(BM, BN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_naive_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

void sgemm_tiled(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(BM, BN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_tiled_kernel<BM, BN, BK><<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

void sgemm_thread_tiled(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_thread_tiled_kernel<BM, BN, BK, TM, TN><<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_naive", &sgemm_naive, "Naive SGEMM");
    m.def("sgemm_tiled", &sgemm_tiled, "Tiled SGEMM");
    m.def("sgemm_thread_tiled", &sgemm_thread_tiled, "Thread-tiled SGEMM");
}
