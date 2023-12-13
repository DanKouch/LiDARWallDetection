/*
* dataFrame.hpp
* 
* ME759 Final Project
* Handles LiDAR data frame descriptors
*/

#ifndef DATA_FRAME_HPP
#define DATA_FRAME_HPP

#include <cstdint>

/**
* Descriptor for LiDAR data frame
*/
typedef struct data_frame_desc_t {
    uint32_t numPoints; // The number of points in the frame
    float *x;           // The array of x-coordinates (numPoints long)
    float *y;           // The array of y-coordinates (numPoints long)
    float *z;           // The array of z-coordinates (numPoints long)
} data_frame_desc_t;

/**
* Populates a LiDAR data frame descriptor, given a pointer to
* a memory-mapped frame's contents.
*
* data - Memory-mapped frame data
* desc - The descriptor to populate
*/
void populateLidarDataFrameDesc(void *data, data_frame_desc_t *desc);

#endif