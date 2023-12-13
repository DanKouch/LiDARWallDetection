/*
* fileHandler.cpp
* 
* ME759 Final Project
* Handles some file IO, including memory mapping
*/

#include "fileHandler.hpp"

#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <cstring>
#include <unistd.h>
#include "configuration.hpp"

#define LISTING_FILE_NAME "frameList.txt"

/**
* Opens the listing file under the specified input file directory
*
* inputFileDir - The path to the input file directory
*/
FILE *getListingFile(const char *inputFileDir) {
    char inputFileListingPath[PATH_MAX];
    strncpy(inputFileListingPath, inputFileDir, PATH_MAX - strlen(LISTING_FILE_NAME) - 2);
    strcat(inputFileListingPath, "/" LISTING_FILE_NAME);

    FILE *outputFile = fopen(inputFileListingPath, "r");
    if(outputFile == NULL) {
        fprintf(stderr, "Error: Could not open input listing file.\n");
    }

    return outputFile;
}

/**
* Memory-maps the file at the specified path, populating
* a memory map descriptor.
*
* filePath - The file to memory map
* desc - The descriptor to populate
*/
int mmap_file(const char *filePath, mmap_descriptor_t *desc) {
    // Open file descriptor
    int fd = open(filePath, O_RDONLY, 0);

    // Stat the file, to get it's size and to verify we can access it
    struct stat statBuf;
    int fstatError = fstat(fd, &statBuf);
    if(fstatError != 0) {
        fprintf(stderr, "Error: Could not open `%s`: %s\n", filePath, std::strerror(errno));
        return -1;
    }

    // Verify the file is a regular file
    if(!S_ISREG(statBuf.st_mode)) {
        fprintf(stderr, "Error: '%s' is not a regular file.\n", filePath);
        return -1;
    }

    // Perform the memory mapping
    desc->size = (long) statBuf.st_size;

    /*
    * NOTE: When I tried running without PROT_WRITE, I was crashing Euler nodes.
    */

#ifdef __NVCC__
    desc->data = mmap(0, desc->size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_LOCKED, fd, 0);
#else
    desc->data = mmap(0, desc->size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
#endif

    if(desc->data == MAP_FAILED) {
        fprintf(stderr, "Error: Could not memory map file '%s': %s\n", filePath, std::strerror(errno));
        return -1;
    }

    // We don't need to keep the file descriptor open
    close(fd);

    return 0;
}

/**
* Unmaps data associated with the provided memory map descriptor
*
* desc - The memory map descriptor to unmap
*/
int unmmap_file(mmap_descriptor_t *desc) {
    int err = munmap(desc->data, desc->size);
    if(err != 0) {
        fprintf(stderr, "Error: Could not unmap memory: %s\n", std::strerror(errno));
        return -1;
    }

    return 0;
}