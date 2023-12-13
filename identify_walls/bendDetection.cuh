/*
* bendDetection.cuh
* 
* ME759 Final Project
* Functionality for GPU bend detection
*/

#ifdef __NVCC__

#ifndef BEND_DETECTION_HPP
#define BEND_DETECTION_HPP

#include "identifyWalls.hpp"

/** 
* Detects bends in a list of points using a convolution-like operation that
* calculates R-squared values on series of points. The width of the
* convolution-like operation in points is inversely preportional to the distance
* of the point away from the sensor. This is due to the fact that the number of
* points per meter of distance is inversely preportional to the distance away
* from the sensor.
*
* For a point to be characterized as a bend, it's associated r-squared value must
* be above R_SQUARED_THRESHOLD, and the distance of the point to the previous point
* must be below DIST_TOLERANCE.

* This function assumes points are in order of their angle theta (i.e., that
* angle increases as index increases, and that the last point is a neihbor of
* the first point), which is true for our LiDAR data.

* pX - The x-coordinatess of the input points
* pY - The y-coordinatess of the input points
* pZ - The z-coordinatess of the input points
* numPoints - The number of points (i.e., the length of pX, pY, and pZ)
* bends - An array populated with boolean values, where a 1 correponds
* to a bend
*/
__global__ void detectBends(const float* __restrict__ pX,
                            const float* __restrict__ pY,
                            const float* __restrict__ pZ,
                            uint32_t numPoints,
                            uint8_t* __restrict__ bends);

#endif

#endif