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
#include "zeroDegree.hpp"

using namespace std;

#define USAGE_STRING "./planeExtraction [outputFile] [inputFileDirectory] [numFrames]"

#define MAX_SEGMENT_DESC 500

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
    planeExtract(d_px, d_py, d_pz, dataFrameDesc.numPoints, segmentDescs, &numSegmentDesc);
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
    cpuPlaneExtract(&dataFrameDesc, segmentDescs, MAX_SEGMENT_DESC, &numSegmentDesc);
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

int main(int argc, char **argv) {
    if(argc != 4) {
        fprintf(stderr, "Error: Invalid usage.\n%s\n", USAGE_STRING);
        return -1;
    }

    char *outputFileName = argv[1];
    char *inputFileDir = argv[2];

    int numFrames = (int) atoi(argv[3]);
    if(numFrames < 0) {
        fprintf(stderr, "Usage Error: numFrames can't be negative.");
        fprintf(stderr, "%s\n", USAGE_STRING);
    }

    printf("NumFrames: %d\n", numFrames);

    // Open output file
    FILE *outputFile = fopen(outputFileName, "w");
    if(outputFile == NULL) {
        fprintf(stderr, "Error: Could not open/create output file '%s'\n", outputFileName);
    }


    FILE *listingFd = getListingFile(inputFileDir);

    printf("Input file dir: %s\n", inputFileDir);

    char inputFilePath[500];
    int framesCompleted = 0;

    // Allocate working memory for plane extraction
    void *d_points = NULL;
    segment_desc_t *segmentDescs;
#ifdef __NVCC__
    CHECK_CUDA(cudaMalloc((void **) &d_points, sizeof(float) * MAX_POINTS * 3));
    CHECK_CUDA(cudaMallocManaged((void **) &segmentDescs, sizeof(segment_desc_t) * MAX_SEGMENTS, cudaMemAttachGlobal));
    planeExtractAllocateTempMem();
#else
    segmentDescs = (segment_desc_t *) malloc(sizeof(segmentDescs) * MAX_SEGMENT_DESC);
#endif

    chrono::high_resolution_clock::time_point start;
    chrono::high_resolution_clock::time_point end;

    start = chrono::high_resolution_clock::now();
    char fileName[BIN_FILE_MAX_LENGTH];
    while (fgets(fileName, BIN_FILE_MAX_LENGTH, listingFd) != NULL) {
        // Get rid of new line character
        fileName[strlen(fileName) - 1] = 0;

        getInputFilePath(fileName, inputFileDir, inputFilePath);

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
    planeExtractFreeTempMem();
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

