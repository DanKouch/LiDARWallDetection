/*
* segmentMerging.cuh
* 
* ME759 Final Project
* Functionality for GPU segment merging
*/

#ifdef __NVCC__

#ifndef SEGMENT_MERGING_HPP
#define SEGMENT_MERGING_HPP

#include "identifyWalls.hpp"

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
                                        const float* __restrict__ pZ);

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
                                  const float *pZ);

#endif

#endif