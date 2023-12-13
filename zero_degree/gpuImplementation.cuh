#ifdef __NVCC__

#ifndef GPU_IMPLEMENTATION_CUH
#define GPU_IMPLEMENTATION_CUH

#include "zeroDegree.hpp"

void planeExtractAllocateTempMem();
void planeExtractFreeTempMem();
int planeExtract(float *pX, float *pY, float *pZ, uint32_t numPoints, segment_desc_t *segmentDescs, uint32_t *numSegmentDesc);

#endif

#endif