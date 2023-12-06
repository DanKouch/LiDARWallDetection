#ifndef FILE_HANDLER_HPP
#define FILE_HANDLER_HPP

typedef struct mmap_descriptor_t {
    long size;
    void *data;
} mmap_descriptor_t;

int mmap_file(const char *filePath, mmap_descriptor_t *desc);
int unmmap_file(mmap_descriptor_t *desc);

#endif