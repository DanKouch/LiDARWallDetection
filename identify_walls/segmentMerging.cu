/*
* segmentMerging.cu
* 
* ME759 Final Project
* Functionality for GPU segment merging
*/

#ifdef __NVCC__

#include <cuda.h>

#include "segmentMerging.cuh"
#include "identifyWalls.hpp"
#include "configuration.hpp"
#include "cudaUtil.cuh"

/**
* Merges all even-or-odd (determined by template variable) indexed segment with the
* previous segment, so long as as the difference in angles of the segments is below
* MERGE_ABS_COS_TOLERANCE and the distance between the segments is below
* DIST_TOLERANCE. If there is no previous segment, the kernel will wrap-around
* and access the last segment.
*
* Note: This kernel is intended to be run in a single block with >= numSegments
* threads.
* 
* segmentDescs - The segment descriptors to merge
* numSegments - The number of segment descriptors
* removedCount - A pointer that gets populated with the number of segments that got
*                removed.
* pX - The x-coordinatess of the input points
* pY - The y-coordinatess of the input points
* pZ - The z-coordinatess of the input points
*/
template<bool odd>
__global__ void mergeNeighboringSegments(segment_desc_t *segmentDescs,
                                        uint32_t numSegments,
                                        uint32_t *removedCount,
                                        const float* __restrict__ pX,
                                        const float* __restrict__ pY,
                                        const float* __restrict__ pZ) {
    
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

                // Keep track of how many segments have been removed
                atomicAdd(removedCount, 1);
            }

        }
    }
}

/**
* Does a single pass of segment merging, merging each segment with its
* immediate neighbors.
*
* segmentDescs - The segment descriptors to merge
* numSegments - The number of segment descriptors
* pX - The x-coordinatess of the input points
* pY - The y-coordinatess of the input points
* pZ - The z-coordinatess of the input points
*/
uint32_t mergeNeighboringSegments(segment_desc_t *segmentDescs,
                                  uint32_t *d_numSegments,
                                  const float *pX,
                                  const float *pY,
                                  const float *pZ) {
    uint32_t numOrigSegments = *d_numSegments;

    uint32_t *numRemoved;
    CHECK_CUDA(cudaMallocManaged((void **) &numRemoved, sizeof(uint32_t), cudaMemAttachGlobal));
    *numRemoved = 0;

    // Merge even-indexed segments with their previous segments
    mergeNeighboringSegments<false><<<1, ((numOrigSegments/2) + 1)>>>(segmentDescs, numOrigSegments, numRemoved, pX, pY, pZ);
    CHECK_CUDA(cudaPeekAtLastError());

    // Merge odd-indexed segments with their previous segments
    mergeNeighboringSegments<true><<<1, ((numOrigSegments/2) + 1)>>>(segmentDescs, numOrigSegments, numRemoved, pX, pY, pZ);
    CHECK_CUDA(cudaPeekAtLastError());

    CHECK_CUDA(cudaDeviceSynchronize());

    uint32_t totalRemoved = *numRemoved;

    CHECK_CUDA(cudaFree(numRemoved));

    return totalRemoved;
}

#endif