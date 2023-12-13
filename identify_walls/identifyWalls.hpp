#ifndef IDENTIFY_WALLS_HPP
#define IDENTIFY_WALLS_HPP

#include <cstdint>
#include <cstdio>

/**
Defines a line segment (i.e., wall)
*/
typedef struct segment_desc_t {
    uint32_t segmentStart; // Point index of start of segment
    uint32_t segmentEnd;   // Point index of end of segment
} segment_desc_t;

/**
* Processes a frame of input located at frameFilePath, identifying walls and
* printing out identified walls in the provided output file as CSV.
*
* frameFilePath - path of input file (.bin file)
* frameName - Name of frame (can be just the file's name)
* outputFile - File descriptor for CSV output
* d_points - If compiled with nvcc, this should contain a pointer to an allocated space for point data. Otherwise can be NULL
* segmentDescs - An allocated space to store segment descriptors
*/
int processFrame(char *frameFilePath, char *frameName, FILE *outputFile, void *d_points, segment_desc_t *segmentDescs);

#endif