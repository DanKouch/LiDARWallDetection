/*
* cpuImplementation.hpp
* 
* ME759 Final Project
* CPU (single core) implementation of wall detection.
*/

#ifndef __NVCC__

#include <cstdio>
#include <cstdint>
#include <assert.h>
#include <math.h>

#include "configuration.hpp"
#include "dataFrame.hpp"
#include "identifyWalls.hpp"
#include "cpuImplementation.hpp"


/**
* Identifies straight segments over all points in a r-squared convolution,
* just like the GPU implementation. Unlike the GPU implementation, this
* function doesn't identify bend segments, and filters segments by length,
* and puts segments directly into segment descriptors.
*
* desc - The data frame descriptor to get point data from
* segmentDescOut - The segment descriptor array to populate
* maxSegmentDesc - The maximum number of segments to identify
* numSegmentDesc - A pointer to be populated with the number of identified
*                  segments.
*/
void identifyStraightSegments(const data_frame_desc_t *desc, segment_desc_t segmentDescOut[], uint32_t maxSegmentDesc, uint32_t *numSegmentDesc) {

    uint32_t curSegmentLength = 0;
    uint32_t curSegmentStart = 0;
    uint32_t numSegments = 0;

    for(long i = 0; i < desc->numPoints; i++) {
        
        double sumXY = 0;
        double sumX = 0;
        double sumY = 0;
        double sumXSquared = 0;
        double sumYSquared = 0;

        // Determine convoltion width n, which should be inversely preportional to radius
        double radius = sqrt(pow(desc->x[i], 2) + pow(desc->y[i], 2));
        long n = (long) (((double) REG_POINTS_PER_INV_METER) * (1/radius));

        // Limit the number of points that can be involved in the convolution
        if(n > REG_MAX_CONV_POINTS)
            n = REG_MAX_CONV_POINTS;

        // Ensure n is odd
        if(n % 2 == 0)
            n += 1;
        
        // Perform convolution-like step
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

        // R squared formula:
        // - https://www.got-it.ai/solutions/excel-chat/excel-tutorial/r-squared/r-squared-in-excel
        double r_squared = pow(n*sumXY - sumX*sumY, 2)
                    / ((n*sumXSquared - pow(sumX, 2))
                        * (n*sumYSquared - pow(sumY, 2)));

#ifdef PRINT_R_SQUARED
        printf("%f, %f, %f, %f, %ld\n", desc->x[i], desc->y[i], desc->z[i], r_squared, n);
#endif

        // Distance from previous point
        float dist = 0;
        if(i > 0) {
            dist = sqrt(pow((desc->x[i] - desc->x[i-1]), 2) + pow((desc->y[i] - desc->y[i-1]), 2));
        }

        // Search for straight segments, where r_squared >= R_SQUARED_THRESHOLD
        if(r_squared >= R_SQUARED_THRESHOLD && dist < DIST_TOLERANCE) {
            // If we are at the beginning of a segment
            if(curSegmentLength == 0)
                curSegmentStart = i;
            curSegmentLength++;
        } else if(curSegmentLength != 0){
            // We've just ended a segment
            if(curSegmentLength >= MIN_SEGMENT_LENGTH) {
                uint32_t curSegmentEnd = (uint32_t) (i - 1);

                // Make sure we can fit the segment
                if(numSegments >= maxSegmentDesc) {
                    fprintf(stderr, "Warning: Couldn't fit all segments in %u segment limit.\n", maxSegmentDesc);
                    *numSegmentDesc = numSegments;
                    return;
                }

                // Populate segment descriptor
                segmentDescOut[numSegments].segmentStart = curSegmentStart;
                segmentDescOut[numSegments].segmentEnd = curSegmentEnd;

                numSegments ++;
            }
            curSegmentLength = 0;
        }
    }

    *numSegmentDesc = numSegments;
}

/**
* Merges each segment with their immediate neighbors, just like the GPU
* implementations mergeNeighboringSegments.
*
* desc - The data frame descriptor
* segmentDesc - The list of segment descriptors
* numSegmentDesc - The number of segment descriptors to merge
*
* Returns the number of removed segments.
*/
int mergeNeighboringSegments(const data_frame_desc_t *desc, segment_desc_t segmentDesc[], uint32_t numSegmentDesc) {
    // Don't do anything if there aren't multiple lines
    if(numSegmentDesc <= 1) {
        return 0;
    }

    uint32_t removed = 0;

    for(uint32_t i = 0; i < numSegmentDesc; i++) {
        uint32_t n = (i + 1) >= numSegmentDesc ? 0 : i + 1;

        float x1 = desc->x[segmentDesc[i].segmentStart];
        float y1 = desc->y[segmentDesc[i].segmentStart];
        float x2 = desc->x[segmentDesc[i].segmentEnd];
        float y2 = desc->y[segmentDesc[i].segmentEnd];

        float x3 = desc->x[segmentDesc[n].segmentStart];
        float y3 = desc->y[segmentDesc[n].segmentStart];
        float x4 = desc->x[segmentDesc[n].segmentEnd];
        float y4 = desc->y[segmentDesc[n].segmentEnd];

        // Take dot product of (curEnd-curStart) and (nextEnd-nextStart)
        float dot = (x2-x1)*(x4-x3) + (y2-y1)*(y4-y3);

        // Equivilant to abs(cos(theta)), where theta is angle between the current segment and the next
        float absCos = abs(dot/(
                         sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2)) *
                         sqrt(pow((x4 - x3), 2) + pow((y4 - y3), 2))));
        
        // Take distance between the two segments
        float dist = sqrt(pow((x2 - x3), 2) + pow((y2 - y3), 2));

        // Condition for merging
        if(absCos > MERGE_ABS_COS_TOLERANCE && dist < DIST_TOLERANCE) {
            // Combine current segment with next
            segmentDesc[i].segmentEnd = segmentDesc[n].segmentEnd;

            // Remove next segment by setting its length to 1
            segmentDesc[n].segmentStart = segmentDesc[n].segmentEnd;

            removed ++;

            // Skip next segment
            i++;
        }
    }

    return removed;
}

/**
* Condenses a list of segment descriptors by removing those which have been
* marked for removal by having their length set to 1.
*
* segmentDesc - The segment descriptors to condense
* numSegmentDesc - A pointer that contains the number of segment descriptors to
*                  condense, that then gets populated with the new number of
*                  segment descriptors after the array has been condensed.
*/
void condenseSegments(segment_desc_t segmentDesc[], uint32_t *numSegmentDesc) {
    // Index in condensed array, always copying backwards in the array
    // if at all.
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

/**
* Filters segments by length, marking those for removal which have a length
* in meters below MIN_FINAL_SEGMENT_LENGTH_M. Segments are marked for removal
* by having their length set to 1.
*
* desc - The data frame descriptor
* segmentDesc - The list of segment descriptors
* numSegmentDesc - The number of segment descriptors to filter
*/
void filterSegmentsByLength(const data_frame_desc_t *desc, segment_desc_t segmentDesc[], uint32_t numSegmentDesc) {
    for(uint32_t i = 0; i < numSegmentDesc; i++) {
        float x1 = desc->x[segmentDesc[i].segmentStart];
        float y1 = desc->y[segmentDesc[i].segmentStart];
        float x2 = desc->x[segmentDesc[i].segmentEnd];
        float y2 = desc->y[segmentDesc[i].segmentEnd];

        float length = sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));

        if(length < MIN_FINAL_SEGMENT_LENGTH_M) {
            segmentDesc[i].segmentEnd = segmentDesc[i].segmentStart;
        }
    }
}

/**
* CPU implementation of identifyWalls, which does the same thing as
* the GPU implementation's function.
*
* desc - The data frame descriptor
* segmentDesc - The list of segment descriptors
* maxSegmentDesc - The maximum number of segment descriptors
* numSegmentDesc - A pointer populated with the number of identified
*                  segment descriptors.
*/
int cpuIdentifyWalls(data_frame_desc_t *desc, segment_desc_t *segmentDescs, uint32_t maxSegmentDesc, uint32_t *numSegmentDesc) {

    // 1. Identify sraight segments by using linear regression in a convolution-like fashion,
    //    then extracting segments of at least a specified number of poiunts with r_squared values
    //    greater than some threshold.
    identifyStraightSegments(desc, segmentDescs, maxSegmentDesc, numSegmentDesc);

#ifndef SKIP_SEGMENT_MERGING
    // 2. Comparing each segment with its immediate neighbors, merge segments if they
    //    approximately belong to the same line and are within a specified distance of
    //    one-another. Repeat until no more merging occurs. 
    while(mergeNeighboringSegments(desc, segmentDescs, *numSegmentDesc) > 0) {

        // 3. Condense segment descriptors to remove entries that have been merged (which
        //    are marked by having length 1)
        condenseSegments(segmentDescs, numSegmentDesc);
    }

    // 4. Filter by minimum segment length
    filterSegmentsByLength(desc, segmentDescs, *numSegmentDesc);
    condenseSegments(segmentDescs, numSegmentDesc);
#endif

    return 0;
}

#endif