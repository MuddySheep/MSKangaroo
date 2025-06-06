# Agent Workflow Instructions

## Repository Purpose
This project provides a CUDA-powered demonstration of the SOTA Kangaroo algorithm to solve the secp256k1 discrete logarithm problem.  All code builds a single command-line tool called `rckangaroo`.

## Build
Use the existing Makefile for Linux builds. CUDA 12.0 or higher is expected on the system.

```bash
make      # builds ./rckangaroo
make clean
```

The default NVCC flags target compute capabilities 61, 75, 86 and 89.  If your GPU is older, adjust `NVCCFLAGS` locally rather than committing build changes.

## Testing
No automated tests exist yet.  Before sending a pull request, compile with `make` to ensure no build breakage.  When tests are introduced (see TODOs below), run them with `make test`.

## Coding Conventions
- C++17 for host code and CUDA C++17 for kernels.
- Prefer RAII over raw `cudaMalloc/cudaFree` where possible.
- Every CUDA API call should be followed by an error check macro similar to `CUDA_CHECK_ERROR()`.
- Avoid magic globals: pass values to kernels via parameters or `cudaMemcpyToSymbol`.
- Use `sizeof(var)` rather than literals when copying buffers.
- Place helpers or utilities in new headers rather than expanding monolithic files.

## Open TODOs
1. **Pinned Host Memory** – Replace the heap allocations in `GpuKang.cpp` for GPU transfers with `cudaHostAlloc` and free with `cudaFreeHost`.
2. **Unit Tests** – Add a small `tests/` directory and verify elliptic-curve operations using known secp256k1 vectors.
3. **RAII Wrappers** – Introduce a lightweight RAII class for managing GPU buffers and refactor callers to use it.

## Pull Requests
- Keep diffs focused on the specific change.  Describe how to reproduce any test output in the PR body.
- Always run `make` before committing.  If compilation fails due to missing CUDA, mention it in the PR.

