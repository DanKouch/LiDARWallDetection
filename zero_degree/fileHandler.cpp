#include "fileHandler.hpp"

#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <cstring>
#include <unistd.h>

int mmap_file(const char *filePath, mmap_descriptor_t *desc) {
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
    desc->data = mmap(0, desc->size, PROT_READ, MAP_PRIVATE, fd, 0);

    if(desc->data == MAP_FAILED) {
        fprintf(stderr, "Error: Could not memory map file '%s': %s\n", filePath, std::strerror(errno));
        return -1;
    }

    // We don't need to keep the file descriptor open
    close(fd);

    return 0;
}

int unmmap_file(mmap_descriptor_t *desc) {
    int err = munmap(desc->data, desc->size);
    if(err != 0) {
        fprintf(stderr, "Error: Could not unmap memory: %s\n", std::strerror(errno));
        return -1;
    }

    return 0;
}