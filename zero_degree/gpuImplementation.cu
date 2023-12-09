#ifndef CPU_IMPLEMENTATION

#include <cstdio>
#include <assert.h>
#include <cuda.h>

#include "zeroDegree.hpp"
#include "configuration.hpp"

#define DEBUG_KERNEL
#include "cudaUtil.cuh"

#define THREADS_PER_BLOCK 128

#define INTEGER_DIV_CEIL(A, B) ((A + (B-1)) / B)

__global__ void detectBends(const float* __restrict__ pX, const float* __restrict__ pY, const float* __restrict__ pZ, uint32_t numPoints, uint8_t* __restrict__ bends) {
    __shared__ float shared[(THREADS_PER_BLOCK + REG_MAX_CONV_POINTS) * 3];

    // int dbg_blk = 0;

    float * const s_x = &shared[0];
    float * const s_y = &shared[THREADS_PER_BLOCK + REG_MAX_CONV_POINTS];
    float * const s_z = &shared[(THREADS_PER_BLOCK + REG_MAX_CONV_POINTS) * 2];

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int beginningOfBlockIdx = (blockIdx.x * blockDim.x);

    int pad = REG_MAX_CONV_POINTS/2;

    // Step 1 - Copy relevant points into shared memory
    // Wrap around when we reach the ends of the point array, rather
    // than filling in with zeroes

    // Fill in front padding with first pad threads
    if(threadIdx.x < pad) {
        int padIdx = beginningOfBlockIdx - pad + (int)threadIdx.x;

        if(padIdx < 0)
            padIdx += numPoints;

        // if(blockIdx.x == dbg_blk)
        //     KERN_DBG("Filling front padding [%d] with index %d\n", idx, padIdx);
        s_x[threadIdx.x] = pX[padIdx];
        s_y[threadIdx.x] = pY[padIdx];
        s_z[threadIdx.x] = pZ[padIdx];
    }
    
    // Fill in middle with all threads
    int fillIdx = idx >= numPoints ? idx - numPoints : idx;
    // if(blockIdx.x == dbg_blk)
    //     KERN_DBG("Filling [%d] with index %d\n", threadIdx.x + pad, fillIdx);
    s_x[threadIdx.x + pad] = pX[fillIdx];
    s_y[threadIdx.x + pad] = pY[fillIdx];
    s_z[threadIdx.x + pad] = pZ[fillIdx];

    // Fill in back padding with last pad threads
    if(threadIdx.x >= blockDim.x - pad) {
        int padIdx = idx + pad;

        if(padIdx >= numPoints)
            padIdx -= numPoints;

        // if(blockIdx.x == dbg_blk)
        //     KERN_DBG("Filling back [%d] with index %d\n", threadIdx.x + 2*pad, padIdx);

        s_x[threadIdx.x + 2*pad] = pX[padIdx];
        s_y[threadIdx.x + 2*pad] = pY[padIdx];
        s_z[threadIdx.x + 2*pad] = pZ[padIdx];
    }

    __syncthreads();

    // TODO: Perform r-squared convolution
    bends[idx] = 0;
}

int planeExtract(float *pX, float *pY, float *pZ, uint32_t numPoints, segment_desc_t **segmentDescs, uint32_t *numSegmentDesc) {
    // Limit bounds of convolution to neighboring blocks
    assert(REG_MAX_CONV_POINTS/2 <= THREADS_PER_BLOCK);

    unsigned int num_blocks = INTEGER_DIV_CEIL(numPoints, THREADS_PER_BLOCK);

    uint8_t *d_bends;
    CHECK_CUDA(cudaMallocManaged((void **) &d_bends, sizeof(uint8_t) * numPoints, cudaMemAttachGlobal));

    CHECK_CUDA(cudaMallocManaged((void **) segmentDescs, sizeof(segment_desc_t) * 1, cudaMemAttachGlobal));

    detectBends<<<num_blocks, THREADS_PER_BLOCK>>>(pX, pY, pZ, numPoints, d_bends);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    *numSegmentDesc = 0;

    return 0;
}

#endif