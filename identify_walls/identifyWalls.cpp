/*
* identifyWalls.cpu
* 
* ME759 Final Project
* Main file for identifyWalls program.
*/

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <assert.h>
#include <cstdlib>
#include <chrono>

#ifdef __NVCC__
#include <cuda.h>
#include <cuda_profiler_api.h>
#include "cudaUtil.cuh"
#include "gpuImplementation.cuh"
#else
#include "cpuImplementation.hpp"
#endif

#include "configuration.hpp"
#include "fileHandler.hpp"
#include "dataFrame.hpp"
#include "identifyWalls.hpp"

using namespace std;

/*
Command Line Arguments:

- outputFile is a .csv file that gets populated with the detected segments
- inputFileDirectory is the directory that contains the input frame .bin files (as well as frameList.txt)
- numFrames is the number of frames that should be processed (0 means all frames in frameList.txt should be processed)
*/
#define USAGE_STRING "./identifyWalls [outputFile] [inputFileDirectory] [numFrames]"

/**
* Main function (see USAGE_STRING for meaning of arguments)
*/
int main(int argc, char **argv) {
    // Ensure we have the correct number of arguments
    if(argc != 4) {
        fprintf(stderr, "Error: Invalid usage.\n%s\n", USAGE_STRING);
        return -1;
    }

    char *outputFileName = argv[1];
    char *inputFileDir = argv[2];

    // Get the number of frames
    // Non-numeric arguments interpreted as 0
    // If numFrames is zero, then all frames in the directory will be read
    int numFrames = (int) atoi(argv[3]);
    if(numFrames < 0) {
        fprintf(stderr, "Usage Error: numFrames can't be negative.");
        fprintf(stderr, "%s\n", USAGE_STRING);
    }

    // Open output file
    FILE *outputFile = fopen(outputFileName, "w");
    if(outputFile == NULL) {
        fprintf(stderr, "Error: Could not open/create output file '%s'\n", outputFileName);
    }

    // Open the frame listing file, which contains a list of all
    // frame files
    FILE *listingFd = getListingFile(inputFileDir);

    // Allocate working memory for plane extraction
    void *d_points = NULL;
    segment_desc_t *segmentDescs;
#ifdef __NVCC__
    CHECK_CUDA(cudaMalloc((void **) &d_points, sizeof(float) * MAX_POINTS * 3));
    CHECK_CUDA(cudaMallocManaged((void **) &segmentDescs, sizeof(segment_desc_t) * MAX_SEGMENTS, cudaMemAttachGlobal));
    identifyWallsAllocateTempMem();
#else
    segmentDescs = (segment_desc_t *) malloc(sizeof(segmentDescs) * MAX_SEGMENTS);
#endif

    chrono::high_resolution_clock::time_point start;
    chrono::high_resolution_clock::time_point end;

    char inputFilePath[500];
    int framesCompleted = 0;

    // Start timing for all-frame timer
    start = chrono::high_resolution_clock::now();
    char fileName[BIN_FILE_MAX_LENGTH];

    // Loop through each line in the listing file
    while (fgets(fileName, BIN_FILE_MAX_LENGTH, listingFd) != NULL) {
        // Get rid of new line character
        fileName[strlen(fileName) - 1] = 0;

        // Grab the relative path associated with the input file
        getInputFilePath(fileName, inputFileDir, inputFilePath);

        // Process the frame
        processFrame(inputFilePath, fileName, outputFile, d_points, segmentDescs);

        if(++framesCompleted == numFrames && numFrames != 0)
            break;
    }
    end = chrono::high_resolution_clock::now();

    float ms = (chrono::duration_cast<chrono::duration<float, std::milli>> (end-start)).count();

    printf("Took %f ms to process %d frames.\n", ms, framesCompleted);
    printf("ms per Frame: %f, FPS: %f.\n", ms/framesCompleted, framesCompleted/(ms/1000));

#ifdef __NVCC__
    printf("Used GPU implementation.\n");
#else
    printf("Used CPU implementation.\n");
#endif

    // Deallocate working memory for plane extraction
#ifdef __NVCC__
    CHECK_CUDA(cudaFree(segmentDescs));
    identifyWallsFreeTempMem();
    #ifndef USE_UVM_POINT_DATA
        CHECK_CUDA(cudaFree(d_points));
    #endif
#else
    free(segmentDescs);
#endif

    fclose(listingFd);
    fclose(outputFile);

    return 0;
}

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
int processFrame(char *frameFilePath, char *frameName, FILE *outputFile, void *d_points, segment_desc_t *segmentDescs) {
    assert(segmentDescs != NULL);
    #ifdef __NVCC__
        assert(d_points != NULL);
    #endif

    // Memory map input file
    mmap_descriptor_t mmapDesc;
    if(mmap_file((char *) frameFilePath, &mmapDesc) != 0) {
        return -1;
    }

    data_frame_desc_t dataFrameDesc;
    populateLidarDataFrameDesc(mmapDesc.data, &dataFrameDesc);

    uint32_t numSegmentDesc = 0;

    assert(dataFrameDesc.numPoints <= MAX_POINTS);

#ifdef __NVCC__
    #ifdef USE_UVM_POINT_DATA
        CHECK_CUDA(cudaHostRegister(mmapDesc.data, mmapDesc.size, cudaHostRegisterDefault | cudaHostRegisterMapped | cudaHostRegisterReadOnly));
        CHECK_CUDA(cudaHostGetDevicePointer((void**) &d_points, (void*) mmapDesc.data, 0));

        float *d_px = (float *) ((char *) d_points + sizeof(uint32_t));
        float *d_py = (float *) ((char *) d_points + sizeof(uint32_t) + (sizeof(float) * dataFrameDesc.numPoints));
        float *d_pz = (float *) ((char *) d_points + sizeof(uint32_t) + (2 * sizeof(float) * dataFrameDesc.numPoints));
    #else
        assert(dataFrameDesc.numPoints <= MAX_POINTS);
        CHECK_CUDA(cudaMemcpy(d_points, (void *) (((uint32_t *) mmapDesc.data) + 1), sizeof(float) * dataFrameDesc.numPoints * 3, cudaMemcpyHostToDevice));

        float *d_px = (float *) d_points;
        float *d_py = (float *) ((char *) d_points + (sizeof(float) * dataFrameDesc.numPoints));
        float *d_pz = (float *) ((char *) d_points + (2 * sizeof(float) * dataFrameDesc.numPoints));
    #endif

    // Create CUDA timing events
    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    identifyWalls(d_px, d_py, d_pz, dataFrameDesc.numPoints, segmentDescs, &numSegmentDesc);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    #ifdef USE_UVM_POINT_DATA
        CHECK_CUDA(cudaHostUnregister(mmapDesc.data));
    #endif

#else
    chrono::high_resolution_clock::time_point start;
    chrono::high_resolution_clock::time_point end;

    start = chrono::high_resolution_clock::now();
    cpuIdentifyWalls(&dataFrameDesc, segmentDescs, MAX_SEGMENTS, &numSegmentDesc);
    end = chrono::high_resolution_clock::now();

    float ms = (chrono::duration_cast<chrono::duration<float, std::milli>> (end-start)).count();
#endif

    printf("[%s] Took %f ms.\n", frameName, ms);

#ifndef PRINT_R_SQUARED
    #ifdef PRINT_INDICES
        // Print segment indices in csv format
        // segment_start_index, segment_end_index
        for(uint32_t i = 0; i < numSegmentDesc; i++) {
            fprintf(outputFile, "%s, %u, %u\n", frameName, segmentDescs[i].segmentStart, segmentDescs[i].segmentEnd);
        }
    #else
        // Print segments in csv format
        // start_x, start_y, end_x, end_y
        for(uint32_t i = 0; i < numSegmentDesc; i++) {
            uint32_t start = segmentDescs[i].segmentStart;
            uint32_t end = segmentDescs[i].segmentEnd;
            fprintf(outputFile, "%s, %f, %f, %f, %f\n", frameName, dataFrameDesc.x[start], dataFrameDesc.y[start],
                                    dataFrameDesc.x[end], dataFrameDesc.y[end]);
        }
    #endif
#endif

    if(unmmap_file(&mmapDesc) != 0) {
        return -1;
    }

    return 0;
}