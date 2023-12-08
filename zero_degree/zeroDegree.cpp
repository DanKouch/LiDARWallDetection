#include <cstdio>
#include <cstdint>
#include <assert.h>
#include <cstdlib>

#ifndef CPU_IMPLEMENTATION
#include <cuda.h>
#include "cudaUtil.cuh"
#endif

#include "fileHandler.hpp"
#include "dataFrame.hpp"
#include "cpuImplementation.hpp"
#include "zeroDegree.hpp"

using namespace std;

#define USAGE_STRING "./planeExtraction [fileName]"

#define MAX_SEGMENT_DESC 500

int main(int argc, char **argv) {
    if(argc != 2) {
        fprintf(stderr, "Error: Invalid usage.\n%s\n", USAGE_STRING);
        return -1;
    }

    char *fileName = argv[1];
    
    mmap_descriptor_t mmapDesc;
    if(mmap_file(fileName, &mmapDesc) != 0) {
        return -1;
    }

    data_frame_desc_t dataFrameDesc;
    populateLidarDataFrameDesc(mmapDesc.data, &dataFrameDesc);

    segment_desc_t *segmentDescs = (segment_desc_t *) malloc(sizeof(segmentDescs) * MAX_SEGMENT_DESC);
    uint32_t numSegmentDesc = 0;

#ifndef CPU_IMPLEMENTATION
    printf("Hello!\n");

    void *d_data;
    CHECK_CUDA(cudaHostRegister(mmapDesc.data, mmapDesc.size, cudaHostRegisterDefault | cudaHostRegisterMapped | cudaHostRegisterReadOnly));

    printf("Registered!\n");

    CHECK_CUDA(cudaHostGetDevicePointer((void**) &d_data, (void*) mmapDesc.data, 0));

    printf("Got device pointer\n");

    CHECK_CUDA(cudaHostUnregister(mmapDesc.data));
#else
    cpuPlaneExtract(&dataFrameDesc, segmentDescs, MAX_SEGMENT_DESC, &numSegmentDesc);
#endif

#ifndef PRINT_R_SQUARED
#ifdef PRINT_INDICES
    // Print segment indices in csv format
    // segment_start_index, segment_end_index
    for(uint32_t i = 0; i < numSegmentDesc; i++) {
        printf("%u, %u\n", segmentDescs[i].segmentStart, segmentDescs[i].segmentEnd);
    }
#else
    // Print segments in csv format
    // start_x, start_y, end_x, end_y
    for(uint32_t i = 0; i < numSegmentDesc; i++) {
        uint32_t start = segmentDescs[i].segmentStart;
        uint32_t end = segmentDescs[i].segmentEnd;
        printf("%f, %f, %f, %f\n", dataFrameDesc.x[start], dataFrameDesc.y[start],
                                   dataFrameDesc.x[end], dataFrameDesc.y[end]);
    }
#endif
#endif

    free(segmentDescs);

    if(unmmap_file(&mmapDesc) != 0) {
        return -1;
    }
}
