#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <thread>
#include <string>

#include "bpe.h"
#include "ggml-backend.h"

#define BIOGPT_FILE_MAGIC   'ggml'

template<typename T>
static void read_safe(std::ifstream& infile, T& dest) {
    infile.read((char*)& dest, sizeof(T));
}

template<typename T>
static void write_safe(std::ofstream& outfile, T& dest) {
    outfile.write((char*)& dest, sizeof(T));
}

struct biogpt_hparams {
    int32_t n_vocab     = 42384;
    int32_t n_merges    = 40000;
    int32_t d_ff        = 4096;
    int32_t d_model     = 1024;
    int32_t n_layer     = 24;
    int32_t n_head      = 16;
    int32_t n_positions = 1024;

    int32_t ftype       = 0;
};

struct biogpt_vocab {
    using id    = int32_t;
    using token = std::string;

    int n_vocab  = 42384;
    int n_merges = 40000;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;

    std::map<word_pair, int> bpe_ranks;
};

typedef std::vector<biogpt_vocab::id> token_sequence;

struct biogpt_layer_decoder {
    // self-attention
    struct ggml_tensor * q_proj_w;
    struct ggml_tensor * k_proj_w;
    struct ggml_tensor * v_proj_w;
    struct ggml_tensor * o_proj_w;

    struct ggml_tensor * q_proj_b;
    struct ggml_tensor * k_proj_b;
    struct ggml_tensor * v_proj_b;
    struct ggml_tensor * o_proj_b;

    // layer norm
    struct ggml_tensor * ln_0_w;
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_0_b;
    struct ggml_tensor * ln_1_b;

    // feed forward
    struct ggml_tensor * fc_0_w;
    struct ggml_tensor * fc_0_b;
    struct ggml_tensor * fc_1_w;
    struct ggml_tensor * fc_1_b;

};

struct biogpt_model {
    biogpt_hparams hparams;

    struct ggml_tensor * embed_tokens;  // token embeddings
    struct ggml_tensor * embed_pos;     // position embeddings

    // final layer norm
    struct ggml_tensor * ln_w;
    struct ggml_tensor * ln_b;

    // lm head
    struct ggml_tensor * lm_head;

    // key + value memory
    struct ggml_tensor * memory_k;
    struct ggml_tensor * memory_v;

    std::vector<biogpt_layer_decoder> layers_decoder;

    // context
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
    int n_loaded;

    // memory
    ggml_backend_t backend = NULL;
    
    ggml_backend_buffer_t buffer_w;
    ggml_backend_buffer_t buffer_kv;
};

struct biogpt_params {
    int32_t seed      = -1; // RNG seed
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_predict = 200; // new tokens to predict

    // sampling parameters
    int32_t top_k = 40;
    float   top_p = 0.9f;
    float   temp  = 0.9f;

    uint8_t verbosity = 0;  // verbosity level

    int32_t n_batch = 8; // batch size for prompt processing

    std::string model = "../ggml_weights/ggml-model.bin"; // model path
    std::string prompt;
    std::string lang;
};

bool biogpt_model_load(
        const std::string & fname,
             biogpt_model & model,
             biogpt_vocab & vocab,
            const uint8_t   verbosity);

void biogpt_model_quantize_internal(
            std::ifstream & fin,
            std::ofstream & fout,
         const ggml_ftype   ftype);

struct ggml_cgraph * biogpt_graph(
            const biogpt_model & model,
            struct ggml_allocr * allocr, 
          const token_sequence & embed_inp,
                     const int   n_past);

bool biogpt_eval(
       const biogpt_model & model,
     const token_sequence & embed_inp,
       std::vector<float> & logits,
       struct ggml_allocr * allocr,
                const int   n_past,
                const int   n_threads);

token_sequence gpt_tokenize(
             biogpt_vocab & vocab,
        const std::string & text,
        const std::string & lang);

std::string gpt_decode(
 std::vector<std::string> & tokens,
        const std::string & lang);

biogpt_vocab::id biogpt_sample_top_k_top_p(
       const biogpt_vocab & vocab,
              const float * logits,
                      int   top_k,
                   double   top_p,
                   double   temp,
             std::mt19937 & rng);

bool biogpt_params_parse(int argc, char ** argv, biogpt_params & params);

void biogpt_print_usage(char ** argv, const biogpt_params & params);
