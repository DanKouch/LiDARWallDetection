/*
* configuration.hpp
* 
* ME759 Final Project
* Configuration parameters
*/

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

// Configuration parameters

// The maximum number of supported points in a frame
#define MAX_POINTS 4096

// The maximum number of supported segments in a frame
#define MAX_SEGMENTS (MAX_POINTS/2)

// The number of threads per block used in detectBends,
// filterValidSegments, and lengthsAndOffsetsToSegmentDescs kernels.
#define THREADS_PER_BLOCK 128

// Threshold for R-squared convolution step
#define R_SQUARED_THRESHOLD 0.9

// When extracting segments of points below the R-squared threshold,
// what is the minimum segment length that should be considered?
#define MIN_SEGMENT_LENGTH 4

// Factor for how many R-squared convolution points are required per
// inverse meter. This number is multiplied by 1/radius to normalize
// convolution width by linear distance, rather than by number of
// points
#define REG_POINTS_PER_INV_METER 350

// The maximum number of points that can be considered in the R-squared
// convolution step
// MUST BE ODD
#define REG_MAX_CONV_POINTS 255

// When merging segments, the abs(cos(theta)) tolerance for considering
// two segments to have the same angle (1 means that they would need
// to have exactly the same angle)
#define MERGE_ABS_COS_TOLERANCE 0.95

// When merging segments, the tolerance for how close two segments need to
// be in meters.
#define DIST_TOLERANCE 1

// The minimum final segment length, in meters
#define MIN_FINAL_SEGMENT_LENGTH_M 0.5

// The max file name length of bin files
#define BIN_FILE_MAX_LENGTH 64

// Allocated memory for CUB temporary storage
#define CUB_TEMP_STORAGE_SIZE 2048

#endif