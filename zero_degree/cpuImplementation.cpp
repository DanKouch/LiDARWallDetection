#include <cstdio>
#include <cstdint>
#include <assert.h>
#include <math.h>

#include "dataFrame.hpp"
#include "cpuImplementation.hpp"

// Parameters
#define CPU_IMPL_THRESHOLD 0.95
#define CPU_IMPL_MIN_SEGMENT_LENGTH 3
#define CPU_IMPL_REG_POINTS_PER_INV_METER 140
#define CPU_IMPL_REG_MAX_CONV_POINTS 500

// #define PRINT_R_SQUARED_CSV
// #define PRINT_BEND_CSV
// #define SKIP_SEGMENT_MERGING

// R squared formula:
// - https://www.got-it.ai/solutions/excel-chat/excel-tutorial/r-squared/r-squared-in-excel

void identifyStraightSegments(const data_frame_desc_t *desc, segment_desc_t segmentDescOut[], uint32_t maxSegmentDesc, uint32_t *numSegmentDesc) {

    uint32_t curSegmentLength = 0;
    uint32_t curSegmentStart;
    uint32_t numSegments = 0;

    for(long i = 0; i < desc->numPoints; i++) {
        
        double sumXY = 0;
        double sumX = 0;
        double sumY = 0;
        double sumXSquared = 0;
        double sumYSquared = 0;

        // Determine convoltion width n, which should be inversely preportional to radius
        double radius = sqrt(pow(desc->x[i], 2) + pow(desc->x[i], 2));
        long n = (long) (((double) CPU_IMPL_REG_POINTS_PER_INV_METER) * (1/radius));

        // Limit the number of points that can be involved in the convolution
        if(n > CPU_IMPL_REG_MAX_CONV_POINTS)
            n = CPU_IMPL_REG_MAX_CONV_POINTS;

        // Ensure n is odd
        if(n % 2 == 0)
            n += 1;
        
        for(long k = -n/2; k <= n/2; k++) {
            long convI = i + k;

            // Wrap around
            if(convI < 0) {
                convI = convI + desc->numPoints;
            } else if(convI > desc->numPoints - 1) {
                convI = convI - desc->numPoints;
            }

            sumXY += desc->x[convI] * desc->y[convI]; 
            sumX += desc->x[convI];
            sumY += desc->y[convI];
            sumXSquared += pow(desc->x[convI], 2);
            sumYSquared += pow(desc->y[convI], 2);
        }

        double r_squared = pow(n*sumXY - sumX*sumY, 2)
                    / ((n*sumXSquared - pow(sumX, 2))
                        * (n*sumYSquared - pow(sumY, 2)));

#ifdef PRINT_R_SQUARED_CSV
        printf("%f, %f, %f, %f\n", desc->x[i], desc->y[i], desc->z[i], r_squared);
#endif

    // A bend has occured if r_squared < CPU_IMPL_THRESHOLD

#ifdef PRINT_BEND_CSV
        printf("%f, %f, %f, %d\n", desc->x[i], desc->y[i], desc->z[i], (r_squared < CPU_IMPL_THRESHOLD ? 1 : 0));
#endif

        // Search for straight segments
        if(r_squared >= CPU_IMPL_THRESHOLD) {
            // If we are at the beginning of a segment
            if(curSegmentLength == 0)
                curSegmentStart = i;
            curSegmentLength++;
        } else if(curSegmentLength != 0){
            // We've just ended a segment
            // TODO: Determine if it makes more sense to search for segments that are larger than
            //       a specified length in meters, rather than a number of points
            if(curSegmentLength >= CPU_IMPL_MIN_SEGMENT_LENGTH) {
                uint32_t curSegmentEnd = (uint32_t) (i - 1);

                // Make sure we can fit the segment
                if(numSegments >= maxSegmentDesc) {
                    fprintf(stderr, "Warning: Couldn't fit all segments in %u segment limit.\n", maxSegmentDesc);
                    *numSegmentDesc = numSegments;
                    return;
                }

                segmentDescOut[numSegments].segmentStart = curSegmentStart;
                segmentDescOut[numSegments].segmentEnd = curSegmentEnd;

                numSegments ++;
            }
            curSegmentLength = 0;
        }
    }

    *numSegmentDesc = numSegments;
}

// Merge neighboring segments
// Returns number of merged segments
// When two segments are merged, one segment's length is set to 0
int mergeNeighboringSegments(const data_frame_desc_t *desc, segment_desc_t segmentDescOut[], uint32_t numSegmentDesc) {
    return 0;
}

// Remove segments of length 0
// TODO: Determine if this is worth it, compared to just printing the non-zero ones
// TODO: Test to verify this works
void condenseSegments(segment_desc_t segmentDesc[], uint32_t *numSegmentDesc) {
    uint32_t newI = 0;
    for(uint32_t originalI = 0; originalI < *numSegmentDesc; originalI++) {
        uint32_t segmentLength = segmentDesc[originalI].segmentEnd - segmentDesc[originalI].segmentStart;
        
        if(segmentLength > 0) {
            // Only bother copying data if the indicies dont match
            if(newI != originalI) {
                segmentDesc[newI].segmentStart = segmentDesc[originalI].segmentStart;
                segmentDesc[newI].segmentEnd = segmentDesc[originalI].segmentEnd;
            }
            
            newI++;
        }
    }

    *numSegmentDesc = newI;
}

int cpuPlaneExtract(data_frame_desc_t *desc, segment_desc_t *segmentDescs, uint32_t maxSegmentDesc, uint32_t *numSegmentDesc) {
    float *convolutionOut = (float *) malloc(sizeof(float) * desc->numPoints);
    if(convolutionOut == nullptr) {
        fprintf(stderr, "Error: Couldn't malloc.\n");
        return -1;
    }

    // 1. Identify sraight segments by using linear regression in a convolution-like fashion,
    //    then extracting segments of at least a specified number of poiunts with r_squared values
    //    greater than some threshold.
    identifyStraightSegments(desc, segmentDescs, maxSegmentDesc, numSegmentDesc);

#ifndef SKIP_SEGMENT_MERGING

    // 2. Comparing each segment with its immediate neighbors, merge segments if they
    //    approximately belong to the same line and are within a specified distance of
    //    one-another. Repeat until no more merging occurs. 

    // TODO: Implement
    while(mergeNeighboringSegments(desc, segmentDescs, *numSegmentDesc) > 0);
    
    // 3. Condense segment descriptors to remove entries that have been merged (which
    //    are marked by having zero length)
    condenseSegments(segmentDescs, numSegmentDesc);
#endif 
    
    return 0;
}