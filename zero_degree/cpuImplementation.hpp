#ifdef CPU_IMPLEMENTATION

#ifndef CPU_IMPLEMENTATION_HPP
#define CPU_IMPLEMENTATION_HPP

#include "dataFrame.hpp"
#include "zeroDegree.hpp"

int cpuPlaneExtract(data_frame_desc_t *desc, segment_desc_t *segmentDescs, uint32_t maxSegmentDesc, uint32_t *numSegmentDesc);

void identifyStraightSegments(const data_frame_desc_t *desc, segment_desc_t segmentDescOut[], uint32_t maxSegmentDesc, uint32_t *numSegmentDesc);

int mergeNeighboringSegments(const data_frame_desc_t *desc, segment_desc_t segmentDesc[], uint32_t numSegmentDesc);

void condenseSegments(segment_desc_t segmentDesc[], uint32_t *numSegmentDesc);

#endif

#endif