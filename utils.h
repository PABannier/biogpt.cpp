#pragma once

#include <string>
#include <vector>
#include <thread>

struct biogpt_params {
    int32_t seed      = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict = 200; // new tokens to predict

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.9f;
    float   temp  = 0.9f;

    int32_t n_batch = 8; // batch size for prompt processing

    std::string model = "./ggml_weights/ggml-model.bin"; // model path
    std::string prompt;
};

bool biogpt_params_parse(int argc, char ** argv, biogpt_params & params);

void biogpt_print_usage(int argc, char ** argv, const biogpt_params & params);
