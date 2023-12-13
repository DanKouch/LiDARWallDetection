#ifndef FILE_HANDLER_HPP
#define FILE_HANDLER_HPP

#include <cstdio>
#include "configuration.hpp"
#include <cstring>

typedef struct mmap_descriptor_t {
    long size;
    void *data;
} mmap_descriptor_t;

FILE *getListingFile(const char *inputFileDir);

inline void getInputFilePath(char *fileName, char *inputFileDir, char *filePath) {
    strcpy(filePath, inputFileDir);//, PATH_MAX - BIN_FILE_MAX_LENGTH - 2);
    strcat(filePath, "/");
    strcat(filePath, fileName);
}

int mmap_file(const char *filePath, mmap_descriptor_t *desc);
int unmmap_file(mmap_descriptor_t *desc);

#endif