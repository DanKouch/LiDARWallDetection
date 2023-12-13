/*
* bendDetection.cu
* 
* ME759 Final Project
* Functionality for GPU bend detection
*/

#ifdef __NVCC__

#include <cuda.h>

#include "bendDetection.cuh"
#include "configuration.hpp"

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
                            uint8_t* __restrict__ bends) {
    __shared__ float shared[(THREADS_PER_BLOCK + REG_MAX_CONV_POINTS) * 3];

    float * const s_x = &shared[0];
    float * const s_y = &shared[THREADS_PER_BLOCK + REG_MAX_CONV_POINTS];
    float * const s_z = &shared[(THREADS_PER_BLOCK + REG_MAX_CONV_POINTS) * 2];

    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    int beginningOfBlockIdx = (blockIdx.x * blockDim.x);

    int pad = REG_MAX_CONV_POINTS/2;

    // Step 1 - Copy relevant points into shared memory
    // Wrap around when we reach the ends of the point array, rather
    // than filling in with zeroes

    // Fill in front padding with first pad threads
    if(threadIdx.x < pad) {
        int padIdx = beginningOfBlockIdx - pad + (int)threadIdx.x;

        if(padIdx < 0)
            padIdx += numPoints;

        s_x[threadIdx.x] = pX[padIdx];
        s_y[threadIdx.x] = pY[padIdx];
        s_z[threadIdx.x] = pZ[padIdx];
    }
    
    // Fill in middle with all threads
    int fillIdx = idx >= numPoints ? idx - numPoints : idx;

    s_x[threadIdx.x + pad] = pX[fillIdx];
    s_y[threadIdx.x + pad] = pY[fillIdx];
    s_z[threadIdx.x + pad] = pZ[fillIdx];

    // Fill in back padding with last pad threads
    if(threadIdx.x >= blockDim.x - pad) {
        int padIdx = idx + pad;

        if(padIdx >= numPoints)
            padIdx -= numPoints;

        s_x[threadIdx.x + 2*pad] = pX[padIdx];
        s_y[threadIdx.x + 2*pad] = pY[padIdx];
        s_z[threadIdx.x + 2*pad] = pZ[padIdx];
    }

    __syncthreads();

    // Step 2 - Perform r-squared convolution
    float sumXY = 0;
    float sumX = 0;
    float sumY = 0;
    float sumXSquared = 0;
    float sumYSquared = 0;

    // Find the number of points (inversely preportional to distance from sensor)
    float radius = hypotf(s_x[threadIdx.x + pad], s_y[threadIdx.x + pad]);
    int n = (int) (REG_POINTS_PER_INV_METER * (1/radius));

    // We need to limit the number of "convolution" points so we don't jump across blocks
    if(n > REG_MAX_CONV_POINTS)
        n = REG_MAX_CONV_POINTS;

    // Ensure the number of "convolution" points is odd
    if(n % 2 == 0)
        n += 1;

    for(int k = -n/2; k <= n/2; k++) {
        int convI = threadIdx.x + pad + k;

        sumXY += s_x[convI] * s_y[convI];
        sumX += s_x[convI];
        sumY += s_y[convI];
        sumXSquared += s_x[convI] * s_x[convI];
        sumYSquared += s_y[convI] * s_y[convI];
    }

    // R-Squared formula from https://www.wallstreetmojo.com/r-squared-formula/
    float r_squared = ((n*sumXY - sumX*sumY)*(n*sumXY - sumX*sumY))
                    / ((n*sumXSquared - (sumX*sumX))
                    * (n*sumYSquared - (sumY*sumY)));

    // Distance from previous point
    float dist = hypotf(s_x[threadIdx.x + pad] - s_x[threadIdx.x + pad - 1], s_y[threadIdx.x + pad] - s_y[threadIdx.x + pad - 1]);

#ifdef PRINT_R_SQUARED
    printf("%f, %f, %f, %f\n", s_x[threadIdx.x + pad], s_y[threadIdx.x + pad], s_z[threadIdx.x + pad], dist < DIST_TOLERANCE ? r_squared : 0);
#endif

    bends[idx] = r_squared < R_SQUARED_THRESHOLD || dist >= DIST_TOLERANCE;
}

#endif