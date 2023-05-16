#pragma once

#include "ggml.h"

#include <cstdio>
#include <string>

#define BIOGPT_FILE_MAGIC   'ggml'
#define BIOGPT_FILE_VERSION 1

#define BIOGPT_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "BIOGPT_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

template <typename T>
static T checked_mul(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        throw format("overflow multiplying %llu * %llu",
                     (unsigned long long) a, (unsigned long long) b);
    }
    return ret;
}

static size_t calc_tensor_size(const std::vector<uint32_t> & ne, enum ggml_type type) {
    size_t size = ggml_type_size(type);
    for (uint32_t dim : ne) {
        size = checked_mul<size_t>(size, dim);
    }
    return size / ggml_blck_size(type);
}

enum biogpt_ftype {
    BIOGPT_FTYPE_ALL_F32     = 0,
    BIOGPT_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
    BIOGPT_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
    BIOGPT_FTYPE_MOSTLY_Q8_0 = 3,  // except 1d tensors
    BIOGPT_FTYPE_MOSTLY_Q5_0 = 4,  // except 1d tensors
};

struct biogpt_file {
    FILE * fp;
    size_t size;

    biogpt_file(const char * fname, const char * mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            throw fprintf(stderr, "failed to open %s: %s", fname, strerror(errno));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        long ret = std::ftell(fp);
        BIOGPT_ASSERT(ret != -1);
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) {
        int ret = std::fseek(fp, (long) offset, whence);
        BIOGPT_ASSERT(ret == 0); 
    }

    void write_raw(const void * ptr, size_t size) {
        if (size == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, size, 1, fp);
        if (ret != 1) {
            throw fprintf(stderr, "write error: %s", strerror(errno));
        }
    }

    void write_u32(std::uint32_t val) {
        write_raw(&val, sizeof(val));
    }

    void close() {
        std::fclose(fp);
    }
};
