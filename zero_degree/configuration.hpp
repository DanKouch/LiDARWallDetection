#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

// Configuration parameters

// Threshold for R-squared convolution step
#define R_SQUARED_THRESHOLD 0.95

// When extracting segments of points below the R-squared threshold,
// what is the minimum segment length that should be considered?
#define MIN_SEGMENT_LENGTH 3

// Factor for how many R-squared convolution points are required per
// inverse meter. This number is multiplied by 1/radius to normalize
// convolution width by linear distance, rather than by number of
// points
#define REG_POINTS_PER_INV_METER 140

// The maximum number of points that can be considered in the R-squared
// convolution step
#define REG_MAX_CONV_POINTS 500

// When merging segments, the abs(cos(theta)) tolerance for considering
// two segments to have the same angle (1 means that they would need
// to have exactly the same angle)
#define MERGE_ABS_COS_TOLERANCE 0.95

// When merging segments, the tolerance for how close two segments need to
// be in meters.
#define DIST_TOLERANCE 1.5

#endif