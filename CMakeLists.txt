cmake_minimum_required(VERSION 3.18)
project(TinyKernel LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find PyTorch
find_package(Torch REQUIRED)

# CUTLASS headers
set(CUTLASS_INCLUDE ${CMAKE_SOURCE_DIR}/thirdparty/cutlass/include)

add_library(cutlass_gemm SHARED
    kernels/cutlass_gemm/cutlass_gemm.cu
)

target_include_directories(cutlass_gemm PRIVATE
    ${CUTLASS_INCLUDE}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(cutlass_gemm PRIVATE ${TORCH_LIBRARIES})
target_compile_options(cutlass_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        --expt-extended-lambda
        --use_fast_math
    >
)

set_target_properties(cutlass_gemm PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
