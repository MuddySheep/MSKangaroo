#include "GpuBuffer.h"
#include <cassert>
#include <cuda_runtime.h>

int main() {
    GpuBuffer buf;
    cudaError_t err = buf.allocate(16);
    CUDA_CHECK_ERROR(err);
    assert(err == cudaSuccess);
    // buffer freed automatically when going out of scope
    return 0;
}
