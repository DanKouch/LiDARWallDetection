#include <cstdio>
#include <cstdint>
#include <assert.h>
#include "fileHandler.hpp"
#include "dataFrame.hpp"

using namespace std;

#define USAGE_STRING "./planeExtraction [fileName]"

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

    printf("Num Points: %d\n", dataFrameDesc.numPoints);

    // Print all points
    for(uint32_t i = 0; i < dataFrameDesc.numPoints; i++) {
        printf("%f, %f, %f\n", dataFrameDesc.x[i], dataFrameDesc.y[i], dataFrameDesc.z[i]);
    }

    if(unmmap_file(&mmapDesc) != 0) {
        return -1;
    }
}

