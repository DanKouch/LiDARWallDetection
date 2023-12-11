#ifndef CPU_IMPLEMENTATION

#include <cstdio>
#include <assert.h>
#include <cuda.h>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>

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

    // Step 2 - Perform r-squared convolution
    float sumXY = 0;
    float sumX = 0;
    float sumY = 0;
    float sumXSquared = 0;
    float sumYSquared = 0;

    float radius = hypotf(s_x[threadIdx.x + pad], s_y[threadIdx.x + pad]);
    int n = (int) (REG_POINTS_PER_INV_METER * (1/radius));

    if(n > REG_MAX_CONV_POINTS)
        n = REG_MAX_CONV_POINTS;

    // TODO: Check if this is actually necessary
    if(n % 2 == 0)
        n += 1;

    for(int k = -n/2; k <= n/2; k++) {
        int convI = threadIdx.x + pad + k;

        sumXY += s_x[convI] * s_y[convI];
        sumX += s_x[convI];
        sumY += s_y[convI];
        sumXSquared += s_x[convI] * s_x[convI];
        sumYSquared += s_y[convI] * s_y[convI];
    }

    float r_squared = ((n*sumXY - sumX*sumY)*(n*sumXY - sumX*sumY))
                    / ((n*sumXSquared - (sumX*sumX))
                    * (n*sumYSquared - (sumY*sumY)));

#ifdef PRINT_R_SQUARED
    printf("%f, %f, %f, %f\n", s_x[threadIdx.x + pad], s_y[threadIdx.x + pad], s_z[threadIdx.x + pad], r_squared);
#endif

    // Distance from previous point
    float dist = hypotf(s_x[threadIdx.x + pad] - s_x[threadIdx.x + pad - 1], s_y[threadIdx.x + pad] - s_y[threadIdx.x + pad - 1]);

    // if(r_squared < R_SQUARED_THRESHOLD || dist >= DIST_TOLERANCE) {
    //     KERN_DBG("[%d] Bend Detected (R^2=%f, Dist=%f)\n", idx, r_squared, dist);
    // }
    
    bends[idx] = r_squared < R_SQUARED_THRESHOLD || dist >= DIST_TOLERANCE;
}

__global__ void filterValidSegments(uint32_t* __restrict__ lengths, const uint32_t* __restrict__ offsets, const uint8_t* __restrict__ bends, uint32_t numRuns) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(idx < numRuns && (bends[offsets[idx]] || lengths[idx] < MIN_SEGMENT_LENGTH)) {
        lengths[idx] = 0;
    }
}

// Transforms offset and length data to array-of-structures
__global__ void lengthsAndOffsetsToSegmentDescs(uint32_t* lengths, uint32_t* offsets, segment_desc_t *segmentDescs, uint32_t numInitialSegments) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(idx < numInitialSegments) {
        segmentDescs[idx].segmentStart = offsets[idx];
        segmentDescs[idx].segmentEnd = offsets[idx] + lengths[idx];
    }
}

void runLengthEncodeBends(uint8_t *d_bends, uint32_t *d_offsets, uint32_t *d_lengths, uint32_t *d_numSegments, uint32_t numPoints) {
    size_t cub_temp_storage_req = 0;
    cub::DeviceRunLengthEncode::NonTrivialRuns(
            NULL,
            cub_temp_storage_req,
            d_bends,
            d_offsets,
            d_lengths,
            d_numSegments,
            numPoints);

    void *d_cubTempStorage;
    CHECK_CUDA(cudaMallocManaged((void **) &d_cubTempStorage, cub_temp_storage_req, cudaMemAttachGlobal));

    cub::DeviceRunLengthEncode::NonTrivialRuns(
            d_cubTempStorage,
            cub_temp_storage_req,
            d_bends,
            d_offsets,
            d_lengths,
            d_numSegments,
            numPoints);

    CHECK_CUDA(cudaFree(d_cubTempStorage));
}

struct NonZeroSegmentLength
{
    CUB_RUNTIME_FUNCTION __device__ __forceinline__
    void NonZeroLength() {}

    CUB_RUNTIME_FUNCTION __device__ __forceinline__
    bool operator()(const segment_desc_t &a) const {
        return a.segmentStart != a.segmentEnd;
    }
};

void condenseSegments(segment_desc_t *segmentDescs, uint32_t *d_numSegments) {
    NonZeroSegmentLength select_op;

    size_t cub_temp_storage_req = 0;
    cub::DeviceSelect::If(
            NULL,
            cub_temp_storage_req,
            segmentDescs,
            segmentDescs,
            d_numSegments,
            *d_numSegments,
            select_op);

    void *d_cubTempStorage;
    CHECK_CUDA(cudaMalloc((void **) &d_cubTempStorage, cub_temp_storage_req));

    cub::DeviceSelect::If(
            d_cubTempStorage,
            cub_temp_storage_req,
            segmentDescs,
            segmentDescs,
            d_numSegments,
            *d_numSegments,
            select_op);

    CHECK_CUDA(cudaFree(d_cubTempStorage));
}

template<bool odd>
__global__ void mergeNeighboringSegments(segment_desc_t *segmentDescs, uint32_t numSegments, uint32_t *removedCount, const float* __restrict__ pX, const float* __restrict__ pY, const float* __restrict__ pZ) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    int curIdx = (idx * 2) + (odd ? 1 : 0);

    if(curIdx < numSegments) {
        int prevIdx = (curIdx - 1) >= 0 ? curIdx - 1 : numSegments - 1;

        segment_desc_t cur = segmentDescs[curIdx];
        segment_desc_t prev = segmentDescs[prevIdx];

        // TODO: Determine if this test is necessary
        if(cur.segmentEnd != cur.segmentStart && prev.segmentStart != prev.segmentEnd) {

            float x1 = pX[prev.segmentStart];
            float y1 = pY[prev.segmentStart];
            float x2 = pX[prev.segmentEnd];
            float y2 = pY[prev.segmentEnd];

            float x3 = pX[cur.segmentStart];
            float y3 = pY[cur.segmentStart];
            float x4 = pX[cur.segmentEnd];
            float y4 = pY[cur.segmentEnd];

            // Take dot product of (curEnd-curStart) and (nextEnd-nextStart)
            float dot = (x2-x1)*(x4-x3) + (y2-y1)*(y4-y3);

            // Equivilant to abs(cos(theta)), where theta is angle between the current segment and the next
            float absCos = fabsf(dot/(hypotf(x2 - x1, y2 - y1) * hypotf(x4 - x3, y4 - y3)));

            float dist = hypotf(x2 - x3, y2 - y3);

            if(absCos > MERGE_ABS_COS_TOLERANCE && dist < DIST_TOLERANCE) {
                // Combine previous segment with current
                segmentDescs[prevIdx].segmentEnd = cur.segmentEnd;

                // Remove current segment by setting its length to 0
                segmentDescs[curIdx].segmentStart = cur.segmentEnd;

                atomicAdd(removedCount, 1);
            }

        }
    }
}

uint32_t mergeSegments(segment_desc_t *segmentDescs, uint32_t *d_numSegments, const float *pX, const float *pY, const float *pZ) {
    uint32_t numOrigSegments = *d_numSegments;

    uint32_t *numRemoved;
    CHECK_CUDA(cudaMallocManaged((void **) &numRemoved, sizeof(uint32_t), cudaMemAttachGlobal));
    *numRemoved = 0;

    mergeNeighboringSegments<false><<<1, ((numOrigSegments/2) + 1)>>>(segmentDescs, numOrigSegments, numRemoved, pX, pY, pZ);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    mergeNeighboringSegments<true><<<1, ((numOrigSegments/2) + 1)>>>(segmentDescs, numOrigSegments, numRemoved, pX, pY, pZ);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());


    uint32_t totalRemoved = *numRemoved;
    printf("Removed %u!\n", totalRemoved);

    CHECK_CUDA(cudaFree(numRemoved));

    return totalRemoved;
}


int planeExtract(float *pX, float *pY, float *pZ, uint32_t numPoints, segment_desc_t **segmentDescs, uint32_t *numSegmentDesc) {
    // Limit bounds of convolution to neighboring blocks
    assert(REG_MAX_CONV_POINTS/2 <= THREADS_PER_BLOCK);

    unsigned int num_blocks = INTEGER_DIV_CEIL(numPoints, THREADS_PER_BLOCK);

    uint8_t *d_bends;
    CHECK_CUDA(cudaMallocManaged((void **) &d_bends, sizeof(uint8_t) * numPoints, cudaMemAttachGlobal));

    detectBends<<<num_blocks, THREADS_PER_BLOCK>>>(pX, pY, pZ, numPoints, d_bends);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    uint32_t *d_offsets;
    uint32_t *d_lengths;
    uint32_t *d_numSegments;

    CHECK_CUDA(cudaMallocManaged((void **) &d_numSegments, sizeof(uint32_t), cudaMemAttachGlobal));

    // Can get away with allocating numPoints/2 since lengths need to be at least 2
    CHECK_CUDA(cudaMallocManaged((void **) &d_offsets, sizeof(uint32_t) * numPoints/2, cudaMemAttachGlobal));
    CHECK_CUDA(cudaMallocManaged((void **) &d_lengths, sizeof(uint32_t) * numPoints/2, cudaMemAttachGlobal));

    runLengthEncodeBends(d_bends, d_offsets, d_lengths, d_numSegments, numPoints);

    uint32_t numInitialSegments = *d_numSegments;

    num_blocks = INTEGER_DIV_CEIL(numInitialSegments, THREADS_PER_BLOCK);
    filterValidSegments<<<num_blocks, THREADS_PER_BLOCK>>>(d_lengths, d_offsets, d_bends, numInitialSegments);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaFree(d_bends));

    CHECK_CUDA(cudaMallocManaged((void **) segmentDescs, sizeof(segment_desc_t) * numInitialSegments, cudaMemAttachGlobal));

    lengthsAndOffsetsToSegmentDescs<<<num_blocks, THREADS_PER_BLOCK>>>(d_lengths, d_offsets, *segmentDescs, numInitialSegments);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // No longer needed, as their information is now in segmentDescs
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_lengths));

#ifndef SKIP_SEGMENT_MERGING
    // Condense segments so long as merging reduces the number of segments
    do {
        condenseSegments(*segmentDescs, d_numSegments);
    } while(mergeSegments(*segmentDescs, d_numSegments, pX, pY, pZ));
#else
    condenseSegments(*segmentDescs, d_numSegments);
#endif

    CHECK_CUDA(cudaFree(d_numSegments));

    *numSegmentDesc = *d_numSegments;

    return 0;
}

#endif