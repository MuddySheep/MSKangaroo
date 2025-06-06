#pragma once
#include <cstdint>

constexpr bool is_little_endian() {
    const uint16_t x = 1;
    return *reinterpret_cast<const uint8_t*>(&x) == 1;
}

static_assert(is_little_endian(), "Big-endian architectures are not supported");

#define CUDA_CHECK_ERROR(call) do { \
    cudaError_t _e = (call); \
    if(_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
    } \
} while(0)

