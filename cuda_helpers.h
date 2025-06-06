#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <stdio.h>
#include <cuda_runtime.h>

static_assert(cudaSuccess == 0, "cudaSuccess constant unexpected");

#define CUDA_CHECK_ERROR(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        abort(); \
    } \
} while(0)

#endif // CUDA_HELPERS_H
