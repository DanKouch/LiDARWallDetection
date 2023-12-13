#ifdef __NVCC__

#ifndef CUDA_UTIL_CUH
#define CUDA_UTIL_CUH

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cuda.h>

#ifdef DEBUG_KERNEL
#define KERN_DBG(f, ...) printf("[B%d T%d]: " f, blockIdx.x, threadIdx.x, __VA_ARGS__)
#else
#define KERN_DBG(f, ...) (void) 0
#endif

// Used this article: https://leimao.github.io/blog/Proper-CUDA-Error-Checking/
// per @214.
#define CHECK_CUDA(ret) checkCuda((ret), #ret, __FILE__, __LINE__)

inline void checkCuda(cudaError_t ret, const char* const func, const char* const file, const int line) {
    if(ret != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at: %s:%d\n", file, line);
        fprintf(stderr, "%s %s\n", cudaGetErrorString(ret), func);
        std::exit(1);
    }
}

#endif

#endif