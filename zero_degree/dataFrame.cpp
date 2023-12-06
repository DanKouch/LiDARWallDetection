#include "dataFrame.hpp"

void populateLidarDataFrameDesc(void *data, data_frame_desc_t *desc) {
    desc->numPoints = *((uint32_t *) data);
    desc->x = (float *) ((char *) data + sizeof(uint32_t));
    desc->y = (float *) ((char *) data + sizeof(uint32_t) + (sizeof(float) * desc->numPoints));
    desc->z = (float *) ((char *) data + sizeof(uint32_t) + (2 * sizeof(float) * desc->numPoints));
}