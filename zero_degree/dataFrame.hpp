#ifndef DATA_FRAME_HPP
#define DATA_FRAME_HPP

#include <cstdint>

typedef struct data_frame_desc_t {
    uint32_t numPoints;
    float *x;
    float *y;
    float *z;
} data_frame_desc_t;

void populateLidarDataFrameDesc(void *data, data_frame_desc_t *desc);

#endif