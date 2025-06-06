#include "cuda_helpers.h"
int main() {
    CUDA_CHECK_ERROR(cudaSetDevice(9999)); // why: invalid device triggers macro
    return 0; // unreachable if macro aborts
}
