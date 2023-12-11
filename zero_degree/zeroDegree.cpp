#include <cstdio>
#include <cstdint>
#include <assert.h>
#include <cstdlib>

#ifndef CPU_IMPLEMENTATION
#include <cuda.h>
#include "cudaUtil.cuh"
#include "gpuImplementation.cuh"
#else
#include "cpuImplementation.hpp"
#endif

#include "fileHandler.hpp"
#include "dataFrame.hpp"
#include "zeroDegree.hpp"

using namespace std;

#define USAGE_STRING "./planeExtraction [outputFile] [inputFile]"

#define MAX_SEGMENT_DESC 500

int main(int argc, char **argv) {
    if(argc != 3) {
        fprintf(stderr, "Error: Invalid usage.\n%s\n", USAGE_STRING);
        return -1;
    }

    char *outputFileName = argv[1];
    char *inputFileName = argv[2];

    // Open output file
    FILE *outputFile = fopen(outputFileName, "w");
    if(outputFile == NULL) {
        fprintf(stderr, "Error: Could not open/create output file '%s'\n", outputFileName);
    }

    // Memory map input file
    mmap_descriptor_t mmapDesc;
    if(mmap_file(inputFileName, &mmapDesc) != 0) {
        return -1;
    }

    data_frame_desc_t dataFrameDesc;
    populateLidarDataFrameDesc(mmapDesc.data, &dataFrameDesc);

    segment_desc_t *segmentDescs;
    uint32_t numSegmentDesc = 0;

#ifndef CPU_IMPLEMENTATION
    void *d_data;
    CHECK_CUDA(cudaHostRegister(mmapDesc.data, mmapDesc.size, cudaHostRegisterDefault | cudaHostRegisterMapped | cudaHostRegisterReadOnly));
    CHECK_CUDA(cudaHostGetDevicePointer((void**) &d_data, (void*) mmapDesc.data, 0));

    float *d_px = (float *) ((char *) d_data + sizeof(uint32_t));
    float *d_py = (float *) ((char *) d_data + sizeof(uint32_t) + (sizeof(float) * dataFrameDesc.numPoints));
    float *d_pz = (float *) ((char *) d_data + sizeof(uint32_t) + (2 * sizeof(float) * dataFrameDesc.numPoints));

    planeExtract(d_px, d_py, d_pz, dataFrameDesc.numPoints, &segmentDescs, &numSegmentDesc);

    CHECK_CUDA(cudaHostUnregister(mmapDesc.data));
#else
    segmentDescs = (segment_desc_t *) malloc(sizeof(segmentDescs) * MAX_SEGMENT_DESC);
    cpuPlaneExtract(&dataFrameDesc, segmentDescs, MAX_SEGMENT_DESC, &numSegmentDesc);
#endif

#ifndef PRINT_R_SQUARED
#ifdef PRINT_INDICES
    // Print segment indices in csv format
    // segment_start_index, segment_end_index
    for(uint32_t i = 0; i < numSegmentDesc; i++) {
        fprintf(outputFile, "%u, %u\n", segmentDescs[i].segmentStart, segmentDescs[i].segmentEnd);
    }
#else
    // Print segments in csv format
    // start_x, start_y, end_x, end_y
    for(uint32_t i = 0; i < numSegmentDesc; i++) {
        uint32_t start = segmentDescs[i].segmentStart;
        uint32_t end = segmentDescs[i].segmentEnd;
        fprintf(outputFile, "%f, %f, %f, %f\n", dataFrameDesc.x[start], dataFrameDesc.y[start],
                                   dataFrameDesc.x[end], dataFrameDesc.y[end]);
    }
#endif
#endif

#ifndef CPU_IMPLEMENTATION
    CHECK_CUDA(cudaFree(segmentDescs));
#else
    free(segmentDescs);
#endif

    fclose(outputFile);

    if(unmmap_file(&mmapDesc) != 0) {
        return -1;
    }
}

