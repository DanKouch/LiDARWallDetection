/*
* gpuImplementation.cu
* 
* ME759 Final Project
* GPU implementation of identifyWalls
*/

#ifdef __NVCC__

#ifndef GPU_IMPLEMENTATION_CUH
#define GPU_IMPLEMENTATION_CUH

#include "identifyWalls.hpp"

/**
* Allocates globally-defined memory used when identifying walls.
* This memory is reused between frames as an optimization.
*/
void identifyWallsAllocateTempMem();

/**
* Deallocates globally-defined memory used when identifying walls.
* This memory is reused between frames as an optimization.
*/
void identifyWallsFreeTempMem();

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
void identifyWalls(float *pX, float *pY, float *pZ, uint32_t numPoints, segment_desc_t *segmentDescs, uint32_t *numSegmentDesc);

#endif

#endif