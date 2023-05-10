#include "ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>


static const size_t MB = 4*1024*1024;

struct biogpt_vocab {
    using id    = int32_t;
    using token = std::string;

    int n_vocab = 42384;

    std::map<token, id> token_to_id;
    std::map<id, token> id_to_token;
};

// base params for BioGPT
struct biogpt_hparams {
    int32_t n_vocab     = 42384;
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

    // embed tokens
    struct ggml_tensor * embed_tokens;

    // embed positions
    struct ggml_tensor * embed_pos;

    // final layer norm
    struct ggml_tensor * ln_w;
    struct ggml_tensor * ln_b;

    // lm head
    struct ggml_tensor * lm_head;

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

    const ggml_type wtype = model.hparams.f16 ? GGML_TYPE_F16 : GGML_TYPE_F32;

    auto & ctx = model.ctx;

    size_t ctx_size = 0;

    // Evaluating context size
    {
        const auto & hparams = model.hparams;

        const int n_vocab = hparams.n_vocab;
        const int d_ff    = hparams.d_ff;
        const int d_model = hparams.d_model;
        const int n_head  = hparams.n_head;
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

        ctx_size += 4ull*MB; // object overhead

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
        const int n_head  = hparams.n_head;
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


int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model>\n", argv[0]);
        return -1;
    }

    const char * path_model = argv[1];

    biogpt_vocab vocab;
    biogpt_model model;

    if(!biogpt_model_load(path_model, model, vocab)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, path_model);
        return 1;
    }

    ggml_free(model.ctx);
}
