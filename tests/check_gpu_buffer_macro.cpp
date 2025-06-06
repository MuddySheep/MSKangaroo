#include "GpuBuffer.h"
#include <cuda_runtime.h>
int main() {
    // ensures macro expands without redef
    CUDA_CHECK_ERROR(cudaSuccess);
    return 0;
}
