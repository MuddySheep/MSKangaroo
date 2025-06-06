#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include "endian_utils.h" // why: ensure little-endian assumptions

// RAII wrapper around cudaMalloc/cudaFree
class GpuBuffer {
        void* ptr_;
public:
        GpuBuffer() : ptr_(nullptr) {}
        ~GpuBuffer() { reset(); }
        GpuBuffer(const GpuBuffer&) = delete;
        GpuBuffer& operator=(const GpuBuffer&) = delete;
        GpuBuffer(GpuBuffer&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
        GpuBuffer& operator=(GpuBuffer&& other) noexcept { if(this != &other){ reset(); ptr_=other.ptr_; other.ptr_=nullptr; } return *this; }

        cudaError_t allocate(size_t size) {
                cudaError_t e = cudaMalloc(&ptr_, size);
                CUDA_CHECK_ERROR(e);
                return e;
        }
        void reset() {
                if (ptr_) {
                        cudaError_t e = cudaFree(ptr_);
                        CUDA_CHECK_ERROR(e);
                        ptr_ = nullptr;
                }
        }
        template<class T> T* get() const { return reinterpret_cast<T*>(ptr_); }
        void** addr() { return &ptr_; }
};

// host and device must agree on pointer-sized layout
static_assert(sizeof(GpuBuffer) == sizeof(void*), "GpuBuffer layout mismatch");

