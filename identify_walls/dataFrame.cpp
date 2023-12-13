/*
* dataFrame.cpp
* 
* ME759 Final Project
* Handles LiDAR data frame descriptors
*/

#include "dataFrame.hpp"

/**
* Populates a LiDAR data frame descriptor, given a pointer to
* a memory-mapped frame's contents.
*
* data - Memory-mapped frame data
* desc - The descriptor to populate
*/
void populateLidarDataFrameDesc(void *data, data_frame_desc_t *desc) {
    desc->numPoints = *((uint32_t *) data);
    desc->x = (float *) ((char *) data + sizeof(uint32_t));
    desc->y = (float *) ((char *) data + sizeof(uint32_t) + (sizeof(float) * desc->numPoints));
    desc->z = (float *) ((char *) data + sizeof(uint32_t) + (2 * sizeof(float) * desc->numPoints));
}