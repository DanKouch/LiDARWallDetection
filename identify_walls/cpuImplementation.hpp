/*
* cpuImplementation.hpp
* 
* ME759 Final Project
* CPU (single core) implementation of wall detection.
*/

#ifndef __NVCC__

#ifndef CPU_IMPLEMENTATION_HPP
#define CPU_IMPLEMENTATION_HPP

#include "dataFrame.hpp"
#include "identifyWalls.hpp"

/**
* CPU implementation of identifyWalls, which does the same thing as
* the GPU implementation's function.
*
* desc - The data frame descriptor
* segmentDesc - The list of segment descriptors
* maxSegmentDesc - The maximum number of segment descriptors
* numSegmentDesc - A pointer populated with the number of identified
*                  segment descriptors.
*/
int cpuIdentifyWalls(data_frame_desc_t *desc, segment_desc_t *segmentDescs, uint32_t maxSegmentDesc, uint32_t *numSegmentDesc);

#endif

#endif