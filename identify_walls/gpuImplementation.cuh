#ifdef __NVCC__

#ifndef GPU_IMPLEMENTATION_CUH
#define GPU_IMPLEMENTATION_CUH

#include "identifyWalls.hpp"

void identifyWallsAllocateTempMem();
void identifyWallsFreeTempMem();
int identifyWalls(float *pX, float *pY, float *pZ, uint32_t numPoints, segment_desc_t *segmentDescs, uint32_t *numSegmentDesc);

#endif

#endif