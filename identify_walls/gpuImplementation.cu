/*
* gpuImplementation.cu
* 
* ME759 Final Project
* GPU implementation of identifyWalls
*/

#ifdef __NVCC__

#include <cstdio>
#include <assert.h>
#include <cuda.h>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_select.cuh>

#include "identifyWalls.hpp"
#include "configuration.hpp"

#include "segmentMerging.cuh"
#include "bendDetection.cuh"
#include "cudaUtil.cuh"

#define INTEGER_DIV_CEIL(A, B) ((A + (B-1)) / B)

// Globally defined storage that gets reused between runs
uint8_t *d_bends;
uint32_t *d_offsets;
uint32_t *d_lengths;
uint32_t *d_numSegments;

uint32_t *d_cubTempStorage;
size_t cubTempStorageSize;

/**
* Allocates globally-defined memory used when identifying walls.
* This memory is reused between frames as an optimization.
*/
void identifyWallsAllocateTempMem() {
    CHECK_CUDA(cudaMalloc((void **) &d_bends, sizeof(uint8_t) * MAX_POINTS));
    CHECK_CUDA(cudaMallocManaged((void **) &d_numSegments, sizeof(uint32_t), cudaMemAttachGlobal));
    CHECK_CUDA(cudaMalloc((void **) &d_offsets, sizeof(uint32_t) * MAX_SEGMENTS));
    CHECK_CUDA(cudaMalloc((void **) &d_lengths, sizeof(uint32_t) * MAX_SEGMENTS));

    cubTempStorageSize = CUB_TEMP_STORAGE_SIZE;
    CHECK_CUDA(cudaMalloc((void **) &d_cubTempStorage, CUB_TEMP_STORAGE_SIZE));
}

/**
* Deallocated globally-defined memory used when identifying walls.
* This memory is reused between frames as an optimization.
*/
void identifyWallsFreeTempMem() {
    CHECK_CUDA(cudaFree(d_bends));
    CHECK_CUDA(cudaFree(d_numSegments));
    CHECK_CUDA(cudaFree(d_offsets));
    CHECK_CUDA(cudaFree(d_lengths));

    CHECK_CUDA(cudaFree(d_cubTempStorage));
}

/**
* Filters a run-length encoding of a bends array to mark for removal
* runs that are runs of bends or runs that are fewer than
* MIN_SEGMENT_LENGTH points long. Runs are marked for removal by setting
* their lengths to 1.
*
* lengths - Run length encoding lengths
* offsets - Run length encoding offsets
* bends - Bend array (where 1 denotes a bend)
* numRuns - The number of runs (i.e., length of lengths and offsets arrays)
*/
__global__ void filterValidSegments(uint32_t* __restrict__ lengths, const uint32_t* __restrict__ offsets, const uint8_t* __restrict__ bends, uint32_t numRuns) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(lengths[idx] < MIN_SEGMENT_LENGTH || idx < numRuns && (bends[offsets[idx]])) {
        lengths[idx] = 1;
    }
}

/**
* Converts run-length encoding lengths and offsets to segment descriptors
* Assumes segmentDescs is of size MAX_SEGMENTS, and that the number of
* segments is less than or equal to MAX_SEGMENTS (which is checked in the
* calling function).
*
* lengths - Run length encoding lengths
* offsets - Run length encoding offsets
* segmentDescs - Segment descriptor array to populate
* numSegments - The number of segments (i.e., length of lengths and offsets arrays)
*/
__global__ void lengthsAndOffsetsToSegmentDescs(uint32_t* lengths, uint32_t* offsets, segment_desc_t *segmentDescs, uint32_t numSegments) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(idx < numSegments) {
        segmentDescs[idx].segmentStart = offsets[idx];
        segmentDescs[idx].segmentEnd = offsets[idx] + lengths[idx] - 1;
    }
}

/**
* Performs a run-length encoding of a bends array using cub, populating
* the supplied pointers for offsets, lengths, and number of segments.
*
* bends - The bends length to encode
* offsets - A (device) array populated with the offset of each identified segment
* lengths - An (devie) array populated with the the length of each identified segment
* numSegments - A (device) pointer populated with the number of segments identified
* numPoints - The number of points (i.e., the length of the bends array)
*/
void runLengthEncodeBends(uint8_t *bends, uint32_t *offsets, uint32_t *lengths, uint32_t *numSegments, uint32_t numPoints) {
    cub::DeviceRunLengthEncode::NonTrivialRuns(
            d_cubTempStorage,
            cubTempStorageSize,
            bends,
            offsets,
            lengths,
            numSegments,
            numPoints);
}

/**
* cub select_op which selects segments that arent of length 1.
*/
struct NonSingularSegmentLength
{
    CUB_RUNTIME_FUNCTION __device__ __forceinline__
    NonSingularSegmentLength() {}

    CUB_RUNTIME_FUNCTION __device__ __forceinline__
    bool operator()(const segment_desc_t &a) const {
        return a.segmentStart != a.segmentEnd;
    }
};

/**
* Condenses an array of segment descriptors using cub, removing those
* which have been marked for removal by having their length set to 1.
*
* segmentDescs - The (device accessible) segment descriptors to condense
* numSegments - A pointer that contains the number of segment descriptors
*               to condense, then gets set to the new number of segment
*               descriptors.
*/
void condenseSegments(segment_desc_t *segmentDescs, uint32_t *numSegments) {
    NonSingularSegmentLength select_op;

    cub::DeviceSelect::If(
            d_cubTempStorage,
            cubTempStorageSize,
            segmentDescs,
            segmentDescs,
            d_numSegments,
            *numSegments,
            select_op);
}

/*
* GPU kernel that sets the length of segments that are shorter than
* MIN_FINAL_SEGMENT_LENGTH_M to 1, which will then get eliminated by
* a condenseSegments call. Note that this kernel filters by length
* in meters, as opposed to the number of points.
* 
* segmentDescs - The segment descriptors to filter
* numSegments - The number of segments in the segmentDescs array
* pX - The x coordinates of the input points
* pY - The y coordinates of the input points
* pZ - The z coordinates of the input points
*/
__global__ void filterSegmentsByLength(segment_desc_t *segmentDescs, uint32_t numSegments, const float* __restrict__ pX, const float* __restrict__ pY, const float* __restrict__ pZ) {
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    if(idx < numSegments) {
        segment_desc_t seg = segmentDescs[idx];

        float x1 = pX[seg.segmentStart];
        float y1 = pY[seg.segmentStart];
        float x2 = pX[seg.segmentEnd];
        float y2 = pY[seg.segmentEnd];

        if(hypotf(x2 - x1, y2 - y1) < MIN_FINAL_SEGMENT_LENGTH_M) {
            // Mark the segment for removal by setting its length to 0
            segmentDescs[idx].segmentEnd = segmentDescs[idx].segmentStart;
        }
    }
}

/**
* Identifies walls in the provided point arrays, populating the provided
* segmentDescs array.
*
* pX - A device-accessable array containing the x coordinate of each point
* pY - A device-accessable array containing the y coordinate of each point
* pZ - A device-accessable array containing the z coordinate of each point
* numPoints - The number of points (the length of pX, pY, and pZ)
* numSegmentDesc - A pointer that will be populated with the number of segments identified
*/
void identifyWalls(float *pX, float *pY, float *pZ, uint32_t numPoints, segment_desc_t *segmentDescs, uint32_t *numSegmentDesc) {
    // Limit bounds of convolution to neighboring blocks
    assert(REG_MAX_CONV_POINTS/2 <= THREADS_PER_BLOCK);

    unsigned int num_blocks = INTEGER_DIV_CEIL(numPoints, THREADS_PER_BLOCK);
    detectBends<<<num_blocks, THREADS_PER_BLOCK>>>(pX, pY, pZ, numPoints, d_bends);
    CHECK_CUDA(cudaPeekAtLastError());

    // Step 2 - Do Run Length Encoding To Find Sequences of Straights and Bends

    runLengthEncodeBends(d_bends, d_offsets, d_lengths, d_numSegments, numPoints);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    assert(*d_numSegments <= MAX_SEGMENTS);

    // Step 3 - Filter segments to those corresponding to straight lines above a specified length

    num_blocks = INTEGER_DIV_CEIL(*d_numSegments, THREADS_PER_BLOCK);
    filterValidSegments<<<num_blocks, THREADS_PER_BLOCK>>>(d_lengths, d_offsets, d_bends, *d_numSegments);
    CHECK_CUDA(cudaPeekAtLastError());

    // Convert the lengths and offsets arrays into an array of segment_desc_t
    lengthsAndOffsetsToSegmentDescs<<<num_blocks, THREADS_PER_BLOCK>>>(d_lengths, d_offsets, segmentDescs, *d_numSegments);
    
    // Step 4
    // Merge neihboring segments if they are similar enough, condensing between
    // merges that have removed items. Condensing is also done first no matter
    // what, as there will be single-length segments which were filtered out by
    // filterValidSegments

#ifndef SKIP_SEGMENT_MERGING
    // Condense segments so long as merging reduces the number of segments
    do {
        CHECK_CUDA(cudaPeekAtLastError());
        condenseSegments(segmentDescs, d_numSegments);
        CHECK_CUDA(cudaPeekAtLastError());
    } while(mergeNeighboringSegments(segmentDescs, d_numSegments, pX, pY, pZ));

    // Step 5 - Filter segments by minimum length
    filterSegmentsByLength<<<1, *d_numSegments>>>(segmentDescs, *d_numSegments, pX, pY, pZ);
    CHECK_CUDA(cudaPeekAtLastError());
    condenseSegments(segmentDescs, d_numSegments);
    
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
#else
    condenseSegments(segmentDescs, d_numSegments);
    CHECK_CUDA(cudaPeekAtLastError());
#endif

    // Update the caller's value for number of segments
    *numSegmentDesc = *d_numSegments;
}

#endif