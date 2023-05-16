#pragma once

#include "ggml.h"

#include <cstdio>
#include <string>
#include <fstream>
#include <vector>

#define BIOGPT_FILE_MAGIC   'ggml'
#define BIOGPT_FILE_VERSION 1

#define BIOGPT_ASSERT(x) \
    do { \
        if (!(x)) { \
            fprintf(stderr, "BIOGPT_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x); \
            abort(); \
        } \
    } while (0)

using word_pair = std::pair<std::string, std::string>;

static const size_t MB = 4*1024*1024;

template<typename T>
static void read_safe(std::ifstream& infile, T& dest) {
    infile.read((char*)& dest, sizeof(T));
}

template<typename T>
static void write_safe(std::ofstream& outfile, T& dest) {
    outfile.write((char*)& dest, sizeof(T));
}

template <typename T>
static T checked_mul(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        throw fprintf(stderr, "overflow multiplying %llu * %llu",
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
