#ifndef CPU_IMPLEMENTATION_HPP
#define CPU_IMPLEMENTATION_HPP

#include "dataFrame.hpp"

typedef struct segment_desc_t {
    uint32_t segmentStart;
    uint32_t segmentEnd;
} segment_desc_t;

int cpuPlaneExtract(data_frame_desc_t *desc, segment_desc_t *segmentDescs, uint32_t maxSegmentDesc, uint32_t *numSegmentDesc);

#endif