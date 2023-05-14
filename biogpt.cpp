#include "bpe.h"
#include "ggml.h"
#include "mosestokenizer.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <map>
#include <string>
#include <random>
#include <regex>
#include <vector>


static const size_t MB = 4*1024*1024;

struct biogpt_vocab {
    using id    = int32_t;
    using token = std::string;

    int n_vocab  = 42384;
    int n_merges = 40000;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
    std::vector<std::string> special_tokens;

    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    void add_special_token(const std::string &token);
};

// base params for BioGPT
struct biogpt_hparams {
    int32_t n_vocab     = 42384;
    int32_t n_merges    = 40000;
    int32_t d_ff        = 4096;
    int32_t d_model     = 1024;
    int32_t n_layer     = 24;
    int32_t n_head      = 16;
    int32_t f16         = 1;
    int32_t n_positions = 1024;
};

// BioGptDecoderLayer
struct biogpt_layer_decoder {
    // BioGptAttention
    struct ggml_tensor * q_proj_w;
    struct ggml_tensor * k_proj_w;
    struct ggml_tensor * v_proj_w;
    struct ggml_tensor * o_proj_w;

    struct ggml_tensor * q_proj_b;
    struct ggml_tensor * k_proj_b;
    struct ggml_tensor * v_proj_b;
    struct ggml_tensor * o_proj_b;

    // LayerNorm
    struct ggml_tensor * ln_0_w;
    struct ggml_tensor * ln_1_w;
    struct ggml_tensor * ln_0_b;
    struct ggml_tensor * ln_1_b;

    // FF
    struct ggml_tensor * fc_0_w;
    struct ggml_tensor * fc_0_b;
    struct ggml_tensor * fc_1_w;
    struct ggml_tensor * fc_1_b;

};

struct biogpt_model {
    biogpt_hparams hparams;

    struct ggml_tensor * embed_tokens;  // token embeddings
    struct ggml_tensor * embed_pos;  // position embeddings

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
};

template<typename T>
static void read_safe(std::ifstream& infile, T& dest) {
    infile.read((char*)& dest, sizeof(T));
}

static bool biogpt_model_load(const std::string& fname, biogpt_model& model, biogpt_vocab& vocab) {
    fprintf(stderr, "%s: loading model from '%s'\n", __func__, fname.c_str());

    auto infile = std::ifstream(fname, std::ios::binary);
    if (!infile) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic (i.e. ggml signature in hex format)
    {
        uint32_t magic;
        read_safe(infile, magic);
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hyperparams
    {
        auto & hparams = model.hparams;

        read_safe(infile, hparams.n_vocab);
        read_safe(infile, hparams.n_layer);
        read_safe(infile, hparams.n_head);
        read_safe(infile, hparams.n_positions);
        read_safe(infile, hparams.d_ff);
        read_safe(infile, hparams.d_model);
        read_safe(infile, hparams.f16);

        fprintf(stderr, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: d_ff          = %d\n", __func__, hparams.d_ff);
        fprintf(stderr, "%s: d_model       = %d\n", __func__, hparams.d_model);
        fprintf(stderr, "%s: n_positions   = %d\n", __func__, hparams.n_positions);
        fprintf(stderr, "%s: n_head        = %d\n", __func__, hparams.n_head);
        fprintf(stderr, "%s: n_layer       = %d\n", __func__, hparams.n_layer);
        fprintf(stderr, "%s: f16           = %d\n", __func__, hparams.f16);
    }

    // load vocab
    {
        int32_t n_vocab = 0;
        read_safe(infile, n_vocab);

        if(n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            read_safe(infile, len);

            if (len > 0) {
                tmp.resize(len);
                infile.read(&tmp[0], tmp.size()); // read to buffer
                word.assign(&tmp[0], tmp.size());
            } else {
                word = "";
            }

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }

        vocab.n_vocab = model.hparams.n_vocab;

        if (n_vocab < model.hparams.n_vocab) {
            fprintf(stderr, "%s: adding %d extra tokens\n", __func__, model.hparams.n_vocab - n_vocab);
            for (int i = n_vocab; i < model.hparams.n_vocab; i++) {
                word = "[_extra_token_" + std::to_string(i) + "]";
                vocab.token_to_id[word] = i;
                vocab.id_to_token[i] = word;
            }
        }
    }

    // load merges
    {
        int32_t n_merges = 0;
        read_safe(infile, n_merges);

        if(n_merges != model.hparams.n_merges) {
            fprintf(stderr, "%s: invalid model file '%s' (bad merge size %d != %d)\n",
                    __func__, fname.c_str(), n_merges, model.hparams.n_merges);
            return false;
        }

        std::string raw_merge;
        std::pair<std::string, std::string> merge_pair;
        std::vector<char> buf;

        buf.reserve(128);

        for(int i = 0; i < n_merges; i++) {
            uint32_t len;
            read_safe(infile, len);

            if (len > 0) {
                buf.resize(len);
                infile.read(&buf[0], buf.size());
                raw_merge.assign(&buf[0], buf.size());

                // resplit "raw merge" -> ("raw", "merge")
                std::stringstream ss(raw_merge);
                std::string str1, str2;
                ss >> str1 >> str2;

                merge_pair.first  = str1;
                merge_pair.second = str2;
            } else {
                raw_merge = "";
            }

            vocab.bpe_ranks[merge_pair] = i;
        }

        vocab.n_merges = model.hparams.n_merges;
    }

    const ggml_type wtype = model.hparams.f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    // Evaluating context size
    {
        const auto & hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;
        const int d_ff    = hparams.d_ff;
        const int d_model = hparams.d_model;
        const int n_layer = hparams.n_layer;

        ctx_size += n_vocab*d_model*ggml_type_size(wtype);  // lm_head

        // decoder
        {
            ctx_size +=     n_vocab*d_model*ggml_type_size(wtype);         // embed_tokens
            ctx_size += (d_model+2)*d_model*ggml_type_size(wtype);         // embed_pos
            ctx_size +=           2*d_model*ggml_type_size(GGML_TYPE_F32); // final_ln (w and b)
        }

        // decoder layers
        {
            ctx_size += n_layer*(d_model*d_model*ggml_type_size(wtype));  // q_proj_w
            ctx_size += n_layer*(d_model*d_model*ggml_type_size(wtype));  // k_proj_w
            ctx_size += n_layer*(d_model*d_model*ggml_type_size(wtype));  // v_proj_w
            ctx_size += n_layer*(d_model*d_model*ggml_type_size(wtype));  // o_proj_w

            ctx_size += n_layer*(d_model*ggml_type_size(wtype));  // q_proj_b
            ctx_size += n_layer*(d_model*ggml_type_size(wtype));  // k_proj_b
            ctx_size += n_layer*(d_model*ggml_type_size(wtype));  // v_proj_b
            ctx_size += n_layer*(d_model*ggml_type_size(wtype));  // o_proj_b

            ctx_size += 2*n_layer*(d_model*ggml_type_size(GGML_TYPE_F32));  // ln_0_w and ln_0_b
            ctx_size += 2*n_layer*(d_model*ggml_type_size(GGML_TYPE_F32));  // ln_1_w and ln_1_b

            ctx_size += n_layer*(d_ff*d_model*ggml_type_size(wtype)); // wi_0
            ctx_size += n_layer*(d_ff*d_model*ggml_type_size(wtype)); // wi_1

            ctx_size += n_layer*(d_ff*ggml_type_size(wtype));    // bi_0
            ctx_size += n_layer*(d_model*ggml_type_size(wtype)); // bi_1
        }

        ctx_size += 100ull*MB; // object overhead

        fprintf(stderr, "%s: ggml ctx size = %7.2f MB\n", __func__, ctx_size/(1024.0*1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
            .mem_size   = ctx_size,
            .mem_buffer = NULL,
            .no_alloc   = false,
        };

        model.ctx = ggml_init(params);
        if(!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;
        const int d_ff    = hparams.d_ff;
        const int d_model = hparams.d_model;
        const int n_layer = hparams.n_layer;

        model.layers_decoder.resize(n_layer);

        // global
        {
            model.lm_head = ggml_new_tensor_2d(ctx, wtype, d_model, n_vocab);
            model.tensors["output_projection.weight"] = model.lm_head;
        }

        // decoder
        {
            model.embed_tokens = ggml_new_tensor_2d(ctx, wtype, d_model, n_vocab);
            model.embed_pos    = ggml_new_tensor_2d(ctx, wtype, d_model, d_model+2);
            model.ln_w         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);
            model.ln_b         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);

            model.tensors["biogpt.embed_tokens.weight"]     = model.embed_tokens;
            model.tensors["biogpt.embed_positions.weight"]  = model.embed_pos;
            model.tensors["biogpt.layer_norm.weight"]       = model.ln_w;
            model.tensors["biogpt.layer_norm.bias"]         = model.ln_b;

            for (int i = 0; i < n_layer; i++) {
                auto & layer = model.layers_decoder[i];

                layer.q_proj_w = ggml_new_tensor_2d(ctx, wtype, d_model, d_model);
                layer.k_proj_w = ggml_new_tensor_2d(ctx, wtype, d_model, d_model);
                layer.v_proj_w = ggml_new_tensor_2d(ctx, wtype, d_model, d_model);
                layer.o_proj_w = ggml_new_tensor_2d(ctx, wtype, d_model, d_model);

                layer.q_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);
                layer.k_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);
                layer.v_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);
                layer.o_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);

                layer.ln_0_w = ggml_new_tensor_1d(ctx, wtype, d_model);
                layer.ln_1_w = ggml_new_tensor_1d(ctx, wtype, d_model);

                layer.ln_0_b = ggml_new_tensor_1d(ctx, wtype, d_model);
                layer.ln_1_b = ggml_new_tensor_1d(ctx, wtype, d_model);

                layer.fc_0_w = ggml_new_tensor_2d(ctx, wtype, d_model, d_ff);
                layer.fc_1_w = ggml_new_tensor_2d(ctx, wtype, d_ff, d_model);

                layer.fc_0_b = ggml_new_tensor_1d(ctx, wtype, d_ff);
                layer.fc_1_b = ggml_new_tensor_1d(ctx, wtype, d_model);

                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn.q_proj.weight"] = layer.q_proj_w;
                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn.v_proj.weight"] = layer.v_proj_w;
                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn.k_proj.weight"] = layer.k_proj_w;
                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn.out_proj.weight"] = layer.o_proj_w;

                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn.q_proj.bias"] = layer.q_proj_b;
                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn.v_proj.bias"] = layer.v_proj_b;
                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn.k_proj.bias"] = layer.k_proj_b;
                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn.out_proj.bias"] = layer.o_proj_b;

                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn_layer_norm.weight"] = layer.ln_0_w;
                model.tensors["biogpt.layers." + std::to_string(i) + ".self_attn_layer_norm.bias"] = layer.ln_0_b;
                model.tensors["biogpt.layers." + std::to_string(i) + ".final_layer_norm.weight"] = layer.ln_1_w;
                model.tensors["biogpt.layers." + std::to_string(i) + ".final_layer_norm.bias"] = layer.ln_1_b;

                model.tensors["biogpt.layers." + std::to_string(i) + ".fc1.weight"] = layer.fc_0_w;
                model.tensors["biogpt.layers." + std::to_string(i) + ".fc2.weight"] = layer.fc_1_w;

                model.tensors["biogpt.layers." + std::to_string(i) + ".fc1.bias"] = layer.fc_0_b;
                model.tensors["biogpt.layers." + std::to_string(i) + ".fc2.bias"] = layer.fc_1_b;

            }
        }
    }

    // key + value memory
    {
        const auto & hparams  = model.hparams;

        const int d_model     = hparams.d_model;
        const int n_layer     = hparams.n_layer;
        const int n_positions = hparams.n_positions;

        const int n_mem       = n_layer*n_positions;
        const int n_elements  = n_mem*d_model;

        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);

        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

        printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }

    // load weights
    {
        size_t total_size = 0;
        model.n_loaded    = 0;

        while(true) {
            int32_t n_dims;
            int32_t length;
            int32_t ftype;

            read_safe(infile, n_dims);
            read_safe(infile, length);
            read_safe(infile, ftype);

            if (infile.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; i++) {
                read_safe(infile, ne[i]);
                nelements *= ne[i];
            }

            std::string name;
            std::vector<char> buf(length);
            infile.read(&buf[0], buf.size());
            name.assign(&buf[0], buf.size());

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            const size_t element_size = (ftype == 0) ? sizeof(float) :sizeof(ggml_fp16_t);
            if (nelements*element_size != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*element_size);
                return false;
            }

            infile.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            printf("%48s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            model.n_loaded++;
        }

        fprintf(stderr, "%s: model size    = %7.2f MB\n", __func__, total_size/1024.0/1024.0);

        if (model.n_loaded == 0) {
            fprintf(stderr, "%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        } else if (model.n_loaded != (int) model.tensors.size()) {
            fprintf(stderr, "%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
            return false;
        }
    }

    infile.close();

    return true;
}

bool biogpt_eval(
    const biogpt_model& model,
    const int n_threads,
    const int n_past,
    const std::vector<biogpt_vocab::id> & embed_inp,
          std::vector<float>            & logits,
          size_t                        & mem_per_token) {
    const int N = embed_inp.size();

    const auto & hparams = model.hparams;

    const int n_vocab     = hparams.n_vocab;
    const int n_layer     = hparams.n_layer;
    const int n_head      = hparams.n_head;
    const int d_model     = hparams.d_model;
    const int n_positions = hparams.n_positions;

    const int d_kv        = d_model/n_head;

    static size_t buf_size = 256u*1024*1024;
    static void * buf      = malloc(buf_size);

    if (mem_per_token > 0 && mem_per_token*N > buf_size) {
        const size_t buf_size_new = 1.1*(mem_per_token*N);  // add 10% to account for ggml object overhead

        buf_size = buf_size_new;
        buf = realloc(buf, buf_size);
        if (buf == nullptr) {
            fprintf(stderr, "%s: failed to allocate %zu bytes\n", __func__, buf_size);
            return false;
        }
    }

    struct ggml_init_params params = {
        .mem_size   = buf_size,
        .mem_buffer = buf,
        .no_alloc   = false,
    };

    struct ggml_context * ctx0 = ggml_init(params);
    struct ggml_cgraph    gf   = {};
    gf.n_threads = n_threads;

    struct ggml_tensor * embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embed_inp.data(), N*ggml_element_size(embd));

    // token embeddings
    struct ggml_tensor * embed_tokens = ggml_get_rows(ctx0, model.embed_tokens, embd);
    embed_tokens = ggml_scale(ctx0, embed_tokens, ggml_new_f32(ctx0, sqrt(float(d_model))));

    // position embeddings
    struct ggml_tensor * positions = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    for (int i = 0; i < N; ++i) {
        // +2 since BioGPT offsets the embedding ids by 2. specific to biogpt.
        ((int32_t *) positions->data)[i] = n_past + i + 2;
    }
    struct ggml_tensor * embed_positions = ggml_get_rows(ctx0, model.embed_pos, positions);

    // token embeddings + position embeddings
    struct ggml_tensor *inpL = ggml_add(ctx0, embed_tokens, embed_positions);

    for (int layer_ix = 0; layer_ix < n_layer; ++layer_ix) {
        struct ggml_tensor * current;

        // self-attention layer norm
        {
            current = ggml_norm(ctx0, inpL);
            current = ggml_add(
                ctx0,
                ggml_mul(
                    ctx0,
                    ggml_repeat(ctx0, model.layers_decoder[layer_ix].ln_0_w, current), current),
                    ggml_repeat(ctx0, model.layers_decoder[layer_ix].ln_0_b, current)
            );
        }

        // self-attention
        {
            struct ggml_tensor * q_curr = ggml_mul_mat(ctx0, model.layers_decoder[layer_ix].q_proj_w, current);
            q_curr = ggml_add(ctx0, ggml_repeat(ctx0, model.layers_decoder[layer_ix].q_proj_b, q_curr), q_curr);
            q_curr = ggml_reshape_3d(ctx0, q_curr, d_kv, n_head, N);

            // biogpt scales the query
            q_curr = ggml_scale(ctx0, q_curr, ggml_new_f32(ctx0, 1.0f/sqrt(float(d_kv))));

            struct ggml_tensor * k_curr = ggml_mul_mat(ctx0, model.layers_decoder[layer_ix].k_proj_w, current);
            k_curr = ggml_add(ctx0, ggml_repeat(ctx0, model.layers_decoder[layer_ix].k_proj_b, k_curr), k_curr);
            k_curr = ggml_reshape_3d(ctx0, k_curr, d_kv, n_head, N);

            struct ggml_tensor * v_curr = ggml_mul_mat(ctx0, model.layers_decoder[layer_ix].v_proj_w, current);
            v_curr = ggml_add(ctx0, ggml_repeat(ctx0, model.layers_decoder[layer_ix].v_proj_b, v_curr), v_curr);
            v_curr = ggml_reshape_3d(ctx0, v_curr, d_kv, n_head, N);

            // key + value memory
            if (N >= 1) {
                struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*d_model, (ggml_element_size(model.memory_k)*d_model)*(layer_ix*n_positions + n_past));
                struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*d_model, (ggml_element_size(model.memory_v)*d_model)*(layer_ix*n_positions + n_past));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, k_curr, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, v_curr, v));
            }

            // (d_kv, N, n_head)
            struct ggml_tensor * Q = ggml_permute(ctx0, ggml_cpy(ctx0, q_curr, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_kv, n_head, N)), 0, 2, 1, 3);

            // (d_kv, N + n_past, n_head)
            struct ggml_tensor * K =
                ggml_permute(ctx0,
                        ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, model.memory_k, (n_past + N)*d_model, layer_ix*n_positions*ggml_element_size(model.memory_k)*d_model),
                            d_kv, n_head, n_past + N),
                        0, 2, 1, 3);

            // (N + n_past, N, n_head)
            struct ggml_tensor * QK = ggml_mul_mat(ctx0, K, Q);

            // softmax
            struct ggml_tensor * attn_weights = ggml_soft_max(ctx0, QK);

            // [N + n_past, d_kv, n_head]
            struct ggml_tensor * V_trans =
                ggml_cpy(ctx0,
                        ggml_permute(ctx0,
                            ggml_reshape_3d(ctx0,
                                ggml_view_1d(ctx0, model.memory_v, (n_past + N)*d_model, layer_ix*n_positions*ggml_element_size(model.memory_v)*d_model),
                                d_kv, n_head, n_past + N),
                        1, 2, 0, 3),
                        ggml_new_tensor_3d(ctx0, model.memory_v->type, n_past + N, d_kv, n_head)
            );

            // [d_kv, N, n_head]
            struct ggml_tensor * attn_outputs = ggml_mul_mat(ctx0, V_trans, attn_weights);

            // [d_kv, n_head, N]
            struct ggml_tensor * attn_outputs_merged = ggml_permute(ctx0, attn_outputs, 0, 2, 1, 3);

            // [d_model, N]
            current = ggml_cpy(ctx0, attn_outputs_merged, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, d_model, N));

            // output projection
            current = ggml_mul_mat(ctx0, model.layers_decoder[layer_ix].o_proj_w, current);
            current = ggml_add(ctx0, current, ggml_repeat(ctx0, model.layers_decoder[layer_ix].o_proj_b, current));
        }

        // residual connection
        current = ggml_add(ctx0, current, inpL);

        struct ggml_tensor * inpFF = current;

        // feed forward
        {
            // final layer norm
            current = ggml_norm(ctx0, inpFF);
            current = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers_decoder[layer_ix].ln_1_w, current), current), ggml_repeat(ctx0, model.layers_decoder[layer_ix].ln_1_b, current));

            // fc1
            current = ggml_mul_mat(ctx0, model.layers_decoder[layer_ix].fc_0_w, current);
            current = ggml_add(ctx0, ggml_repeat(ctx0, model.layers_decoder[layer_ix].fc_0_b, current), current);

            // gelu
            current = ggml_gelu(ctx0, current);

            // fc2
            current = ggml_mul_mat(ctx0, model.layers_decoder[layer_ix].fc_1_w, current);
            current = ggml_add(ctx0, ggml_repeat(ctx0, model.layers_decoder[layer_ix].fc_1_b, current), current);
        }

        // residual connection
        inpL = ggml_add(ctx0, current, inpFF);
    }

    // final norm layer
    inpL = ggml_norm(ctx0, inpL);
    inpL = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.ln_w, inpL), inpL), ggml_repeat(ctx0, model.ln_b, inpL));

    // lm head
    inpL = ggml_mul_mat(ctx0, model.lm_head, inpL);

    // run computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute       (ctx0, &gf);

    // return result for just the last token
    logits.resize(n_vocab);
    memcpy(logits.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }

    ggml_free(ctx0);

    return true;
}

void biogpt_vocab::add_special_token(const std::string &token) {
    special_tokens.push_back(token);
}

// Extracted from https://github.com/ggerganov/ggml/blob/master/examples/common.cpp
std::vector<biogpt_vocab::id> gpt_tokenize(
    biogpt_vocab & vocab,
    const std::string  & text,
    const std::string  & lang
) {
    // Moses tokenization
    std::vector<std::string> words = moses_tokenize(text, lang);

    // byte-pair encoding and map to vocabulary
    std::vector<biogpt_vocab::id> tokens;
    tokens.push_back(2);  // </s> to start the sequence.
    for (const auto & word : words) {
        std::string bpe_word = bpe(word, vocab.bpe_ranks);

        std::stringstream ss(bpe_word);
        std::string bpe_token;
        while (ss >> bpe_token) {
            if (vocab.token_to_id.find(bpe_token) != vocab.token_to_id.end()) {
                tokens.push_back(vocab.token_to_id.at(bpe_token));
            } else {
                fprintf(stderr, "%s: unknown token '%s'\n", __func__, bpe_token.data());
            }
        }
    }

    return tokens;
}

std::string gpt_decode(std::vector<std::string>& tokens, const std::string& lang) {
    // remove bpe
    std::transform(tokens.begin(), tokens.end(), tokens.begin(), [](std::string t) {
        t = std::regex_replace(t, std::regex(" "), "");
        t = std::regex_replace(t, std::regex("</w>"), " ");
        t = std::regex_replace(t, std::regex("</s>"), " ");
        return t;
    });

    // join the elements of the vector into a single string
    std::string joined_str;
    for (const auto& token : tokens) {
        joined_str += token;
    }

    // split the joined string into individual tokens
    std::vector<std::string> clean_tokens;
    {
        std::stringstream stream(joined_str);
        std::string token;
        while (stream >> token) {
            clean_tokens.push_back(token);
        }
    }

    // detokenize
    std::string out = moses_detokenize(clean_tokens, lang);

    return out;
}

biogpt_vocab::id biogpt_sample_top_k_top_p(
        const biogpt_vocab & vocab,
        const float * logits,
        int    top_k,
        double top_p,
        double temp,
        std::mt19937 & rng) {
    int n_logits = vocab.id_to_token.size();

    std::vector<std::pair<double, biogpt_vocab::id>> logits_id;
    logits_id.reserve(n_logits);

    {
        const double scale = 1.0/temp;
        for (int i = 0; i < n_logits; ++i) {
            logits_id.push_back(std::make_pair(logits[i]*scale, i));
        }
    }

    // find the top K tokens
    std::partial_sort(
            logits_id.begin(),
            logits_id.begin() + top_k, logits_id.end(),
            [](const std::pair<double, biogpt_vocab::id> & a, const std::pair<double, biogpt_vocab::id> & b) {
        return a.first > b.first;
    });

    logits_id.resize(top_k);

    double maxl = -INFINITY;
    for (const auto & kv : logits_id) {
        maxl = std::max(maxl, kv.first);
    }

    // compute probs for the top K tokens
    std::vector<double> probs;
    probs.reserve(logits_id.size());

    double sum = 0.0;
    for (const auto & kv : logits_id) {
        double p = exp(kv.first - maxl);
        probs.push_back(p);
        sum += p;
    }

    // normalize the probs
    for (auto & p : probs) {
        p /= sum;
    }

    if (top_p < 1.0f) {
        double cumsum = 0.0f;
        for (int i = 0; i < top_k; i++) {
            cumsum += probs[i];
            if (cumsum >= top_p) {
                top_k = i + 1;
                probs.resize(top_k);
                logits_id.resize(top_k);
                break;
            }
        }

        cumsum = 1.0/cumsum;
        for (int i = 0; i < (int) probs.size(); i++) {
            probs[i] *= cumsum;
        }
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    int idx = dist(rng);

    return logits_id[idx].second;
}


int main(int argc, char **argv) {
    const int64_t t_main_start_us = ggml_time_us();

    biogpt_params params;

    if (biogpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if(params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);

    int64_t t_load_us = 0;

    biogpt_vocab vocab;
    biogpt_model model;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if(!biogpt_model_load(params.model, model, vocab)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    params.prompt = "HER2 is a protein. It is a trans-membrane receptor.";
    std::vector<biogpt_vocab::id> embed_inp = gpt_tokenize(vocab, params.prompt, params.lang);

    params.n_predict = std::min(params.n_predict, model.hparams.n_positions - (int) embed_inp.size());

    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, embed_inp.size());
    for (int i = 0; i < std::min(8, (int) embed_inp.size()); i++) {
        printf("%d ", embed_inp[i]);
    }
    printf("\n\n");

    std::vector<biogpt_vocab::id> embed;

    // determine the required inference memory per token
    size_t mem_per_token = 0;
    biogpt_eval(model, params.n_threads, 0, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (int i = embed.size(); i < (int) embed_inp.size() + params.n_predict; i++) {
        // predict
        if (embed.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if(!biogpt_eval(model, params.n_threads, n_past, embed, logits, mem_per_token)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embed.size();
        embed.clear();

        if (i >= (int) embed_inp.size()) {
            // sample next token
            const int   top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp  = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            biogpt_vocab::id id = 0;

            // generation
            {
                const int64_t t_start_sample_us = ggml_time_us();
                id = biogpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);
                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            embed.push_back(id);
        } else {
            for (int k = i; k < (int) embed_inp.size(); k++) {
                embed.push_back(embed_inp[k]);
                if ((int) embed.size() > params.n_batch) {
                    break;
                }
            }
            i += embed.size() - 1;
        }

        std::vector<std::string> tokens;
        for (auto id : embed) {
            tokens.push_back(vocab.id_to_token[id]);
        }
        std::string decoded_word = gpt_decode(tokens, params.lang);
        printf("%s ", decoded_word.c_str());
        fflush(stdout);

        // end of text token
        if (embed.back() == model.hparams.n_vocab) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }


    ggml_free(model.ctx);

    return 0;
}
