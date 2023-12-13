/*
* fileHandler.hpp
* 
* ME759 Final Project
* Handles some file IO, including memory mapping
*/

#ifndef FILE_HANDLER_HPP
#define FILE_HANDLER_HPP

// Set PATH_MAX to 4096 if not set elsewhere
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#include <cstdio>
#include "configuration.hpp"
#include <cstring>

/**
* Descriptor for memory-mapped data
*/
typedef struct mmap_descriptor_t {
    long size;
    void *data;
} mmap_descriptor_t;

/**
* Memory-maps the file at the specified path, populating
* a memory map descriptor.
*
* filePath - The file to memory map
* desc - The descriptor to populate
*/
int mmap_file(const char *filePath, mmap_descriptor_t *desc);

/**
* Unmaps data associated with the provided memory map descriptor
*
* desc - The memory map descriptor to unmap
*/
int unmmap_file(mmap_descriptor_t *desc);

/**
* Gets the input file path, given the input directory and filename.
*
* fileName - Input file name
* inputFileDir - Input file directory
* filePath - (Write accessible) File path to populate
*/
inline void getInputFilePath(char *fileName, char *inputFileDir, char *filePath) {
    strcpy(filePath, inputFileDir);
    strcat(filePath, "/");
    strcat(filePath, fileName);
}

/**
* Opens the listing file under the specified input file directory
*
* inputFileDir - The path to the input file directory
*/
FILE *getListingFile(const char *inputFileDir);

#endif