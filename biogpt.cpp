#include "biogpt-util.h"
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
#include <unordered_map>
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
    struct ggml_tensor * attn_layer_norm_w;
    struct ggml_tensor * ffn_norm_w;
    struct ggml_tensor * attn_layer_norm_b;
    struct ggml_tensor * ffn_norm_b;

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

    std::vector<biogpt_layer_decoder> layers;

    // context
    struct ggml_context * ctx;
    std::map<std::string, struct ggml_tensor *> tensors;
    int n_loaded;
};

struct biogpt_load_tensor {
    std::string name;
    enum ggml_type type = GGML_TYPE_F32;

    std::vector<uint32_t> ne;
    size_t size;

    struct ggml_tensor * ggml_tensor = NULL;
    uint8_t * data;

    biogpt_load_tensor(const std::string & name) : name(name) {}

    void calc_all() {
        calc_ne();
        calc_size();
    }

    void calc_ne() {
        // TODO
    }

    void calc_size() {
        size = calc_tensor_size(ne, type);
    }
};

struct biogpt_load_tensors_map {
    std::vector<biogpt_load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;
};

struct biogpt_file_loader {
    std::ifstream infile;
    biogpt_hparams hparams;
    biogpt_vocab vocab;

    int8_t verbosity;

    biogpt_file_loader(const char * fname, int8_t verbosity, biogpt_load_tensors_map & tensors_map): verbosity(verbosity) {
        infile = std::ifstream(fname, std::ios::binary);
        if (!infile) {
            throw fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname);
        }
        read_magic();
        read_hparams();
        read_vocab();
        read_merges();
        read_tensor_metadata(tensors_map);
    }

    void read_magic() {
        uint32_t magic;
        read_safe(infile, magic);
        if (magic != 0x67676d6c) {
            throw fprintf(stderr, "%s: invalid model file (bad magic)\n", __func__);
        }
    }

    void read_hparams() {
        read_safe(infile, hparams.n_vocab);
        read_safe(infile, hparams.n_layer);
        read_safe(infile, hparams.n_head);
        read_safe(infile, hparams.n_positions);
        read_safe(infile, hparams.d_ff);
        read_safe(infile, hparams.d_model);
        read_safe(infile, hparams.f16);
    }

    void read_vocab() {
        int32_t n_vocab = 0;
        read_safe(infile, n_vocab);

        if(n_vocab != hparams.n_vocab) {
            throw fprintf(stderr, "%s: invalid model file (bad vocab size %d != %d)\n",
                          __func__, n_vocab, hparams.n_vocab);
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

        vocab.n_vocab = hparams.n_vocab;

        if (n_vocab < hparams.n_vocab) {
            fprintf(stderr, "%s: adding %d extra tokens\n", __func__, hparams.n_vocab - n_vocab);
            for (int i = n_vocab; i < hparams.n_vocab; i++) {
                word = "[_extra_token_" + std::to_string(i) + "]";
                vocab.token_to_id[word] = i;
                vocab.id_to_token[i] = word;
            }
        }
    }

    void read_merges() {
        int32_t n_merges = 0;
        read_safe(infile, n_merges);

        if(n_merges != hparams.n_merges) {
            throw fprintf(stderr, "%s: invalid model file (bad merge size %d != %d)\n",
                    __func__, n_merges, hparams.n_merges);
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

        vocab.n_merges = hparams.n_merges;
    }

    void read_tensor_metadata(biogpt_load_tensors_map & tensors_map) {
        size_t total_size = 0;
        size_t n_loaded   = 0;

        while(true) {
            int32_t n_dims, length, ftype;

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

            if (tensors_map.name_to_idx.find(name.data()) == tensors_map.name_to_idx.end()) {
                throw fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
            }

            auto tensor = tensors_map.tensors[tensors_map.name_to_idx[name.data()]];
            if (ggml_nelements(tensor) != nelements) {
                throw fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
            }
            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                throw fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%lld, %lld], expected [%d, %d]\n",
                        __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
            }

            const size_t element_size = (ftype == 0) ? sizeof(float) :sizeof(ggml_fp16_t);
            if (nelements*element_size != ggml_nbytes(tensor)) {
                throw fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements*element_size);
            }

            infile.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            if (verbosity > 0) {
                printf("%48s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            }
            total_size += ggml_nbytes(tensor);
            n_loaded++;
        }

        if (verbosity > 0) {
            fprintf(stderr, "%s: model size    = %7.2f MB\n", __func__, total_size/1024.0/1024.0);
        }

        if (n_loaded == 0) {
            throw fprintf(stderr, "%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
        } else if (n_loaded != (int) tensors.size()) {
            throw fprintf(stderr, "%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
        }
    }
};

struct biogpt_model_loader {
    std::unique_ptr<biogpt_file_loader> file_loader;
    size_t num_ggml_tensors_created = 0;
    struct ggml_context * ggml_ctx = NULL;

    biogpt_model_loader(const std::string& fname_base) {
        file_loader = new biogpt_file_loader(fname_base.c_str());
        for (biogpt_load_tensor & lt : tensors_map.tensors) {
            lt.calc_all();
        }
    }

    void calc_sizes(size_t * ctx_size_p) const {
        *ctx_size_p = 0;
        for (const biogpt_load_tensor & lt : tensors_map.tensors) {
            *ctx_size_p += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            *ctx_size_p += lt.size;
        }
    }

    struct ggml_tensor * get_tensor(const std::string & name, const std::vector<uint32_t> & ne) {
        auto it = tensors_map.name_to_idx.find(name);
        if (it == tensors_map.name_to_idx.end()) {
            throw printf("biogpt.cpp: tensor '%s' is missing from model", name.c_str());
        }

        biogpt_load_tensor & lt = tensors_map.tensors.at(it->second);
        if (lt.ne != ne) {
            throw printf("biogpt.cpp: tensor '%s' has wrong shape; expected %s, got %s",
                          name.c_str(), format_tensor_shape(ne).c_str(), format_tensor_shape(lt.ne).c_str());
        }
        
        return get_tensor_for(lt);
    }

    struct ggml_tensor * get_tensor_for(biogpt_load_tensor & lt) {
        struct ggml_tensor * tensor;
        if (lt.ne.size() == 2) {
            tensor = ggml_new_tensor_2d(ggml_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
        } else {
            BIOGPT_ASSERT(lt.ne_size() == 1);
            tensor = ggml_new_tensor_1d(ggml_ctx, lt.type, lt.ne_at(0));
        }
        ggml_set_name(tensor, lt.name.c_str());
        BIOGPT_ASSERT(lt.ggml_tensor == NULL);
        lt.ggml_tensor = tensor;
        num_ggml_tensors_created++;
        return tensor;
    }

    void done_getting_tensors() const {
        if (num_ggml_tensors_created != tensors_map.tensors.size()) {
            throw std::string("biogpt.cpp: file contained more tensors than expected");
        }
    }

    void load_all_data(biogpt_progress_callback progress_callback, void * progress_callback_user_data) {
        size_t data_size = 0;
        for (const biogpt_load_tensor & lt : tensors_map.tensors) {
            data_size += lt.size;
        }

        size_t done_size = 0;
        for (biogpt_load_tensor & lt : tensors_map.tensors) {
            if (progress_callback) {
                progress_callback((float) done_size / data_size, progress_callback_user_data);
            }
            BIOGPT_ASSERT(lt.ggml_tensor); // unused tensors should have been caught by load_data already
            lt.data = (uint8_t *) lt.ggml_tensor->data;
            load_data_for(lt);
            lt.ggml_tensor->data = lt.data;
            done_size += lt.size;
        }
        if (progress_callback) {
            progress_callback(1.0f, progress_callback_user_data);
        }
    }

    void load_data_for(biogpt_load_tensor & lt) {
        biogpt_file & file = file_loader->file;
        file.seek(lt.file_off, SEEK_SET);
        file.read_raw(lt.data, lt.size);
    }
};

static void biogpt_model_load_internal(
        const std::string& fname, 
        biogpt_model& model, 
        biogpt_vocab& vocab, 
        uint8_t& verbosity,
        biogpt_progress_callback progress_callback,
        void * progress_callback_user_data) {

    std::unique_ptr<biogpt_model_loader> ml(new biogpt_model_loader(fname));

    vocab = std::move(ml->file_loader->vocab);
    model.hparams = ml->file_loader->hparams;

    auto &hparams = model.hparams;

    {
        fprintf(stderr, "%s: n_vocab       = %d\n", __func__, hparams.n_vocab);
        fprintf(stderr, "%s: d_ff          = %d\n", __func__, hparams.d_ff);
        fprintf(stderr, "%s: d_model       = %d\n", __func__, hparams.d_model);
        fprintf(stderr, "%s: n_positions   = %d\n", __func__, hparams.n_positions);
        fprintf(stderr, "%s: n_head        = %d\n", __func__, hparams.n_head);
        fprintf(stderr, "%s: n_layer       = %d\n", __func__, hparams.n_layer);
        fprintf(stderr, "%s: f16           = %d\n", __func__, hparams.f16);
    }

    auto & ctx = model.ctx;

    size_t ctx_size;
    ml->calc_sizes(&ctx_size);
    if (verbosity > 0) {
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
            throw("ggml_init() failed\n");
        }
    }

    // prepare memory for the weights
    {
        const uint32_t n_vocab = hparams.n_vocab;
        const uint32_t d_ff    = hparams.d_ff;
        const uint32_t d_model = hparams.d_model;
        const uint32_t n_layer = hparams.n_layer;

        ml->ggml_ctx = ctx;

        model.embed_tokens = ml->get_tensor("embed_tokens.weight",      {d_model, n_vocab});
        model.embed_pos    = ml->get_tensor("embed_positions.weight",   {d_model, d_model+2});
        model.ln_w         = ml->get_tensor("layer_norm.weight",        {d_model});
        model.ln_b         = ml->get_tensor("layer_norm.bias",          {d_model});
        model.lm_head      = ml->get_tensor("output_projection.weight", {d_model, n_vocab});

        model.layers.resize(n_layer);

        for (uint32_t i = 0; i < n_layer; i++) {
            auto & layer = model.layers[i];

            std::string layers_i = "layers." + std::to_string(i);

            layer.attn_layer_norm_w = ml->get_tensor(layers_i + ".self_attn_layer_norm.weight", {d_model});
            layer.attn_layer_norm_b = ml->get_tensor(layers_i + ".self_attn_layer_norm.bias"  , {d_model});

            layer.q_proj_w = ml->get_tensor(layers_i + ".self_attn.q_proj.weight", {d_model, d_model});
            layer.k_proj_w = ml->get_tensor(layers_i + ".self_attn.k_proj.weight", {d_model, d_model});
            layer.v_proj_w = ml->get_tensor(layers_i + ".self_attn.v_proj.weight", {d_model, d_model});
            layer.o_proj_w = ml->get_tensor(layers_i + ".self_attn.o_proj.weight", {d_model, d_model});

            layer.q_proj_b = ml->get_tensor(layers_i + ".self_attn.q_proj.bias", {d_model});
            layer.k_proj_b = ml->get_tensor(layers_i + ".self_attn.k_proj.bias", {d_model});
            layer.v_proj_b = ml->get_tensor(layers_i + ".self_attn.v_proj.bias", {d_model});
            layer.o_proj_b = ml->get_tensor(layers_i + ".self_attn.o_proj.bias", {d_model});

            layer.ffn_norm_w   = ml->get_tensor(layers_i + ".final_layer_norm.weight", {d_model});
            layer.ffn_norm_b   = ml->get_tensor(layers_i + ".final_layer_norm.bias",   {d_model});

            layer.fc_0_w   = ml->get_tensor(layers_i + "fc1.weight", {d_model, d_ff});
            layer.fc_1_w   = ml->get_tensor(layers_i + "fc1.bias"  , {d_model});
        }
    }

    ml->done_getting_tensors();

    // populate `tensors_by_name`
    for (biogpt_load_tensor & lt : ml->tensors_map.tensors) {
        model.tensors_by_name.emplace_back(lt.name, lt.ggml_tensor);
    }

    ml->load_all_data(progress_callback, progress_callback_user_data);

    // model.mapping = std::move(ml->mapping);
}

static bool kv_cache_init(
        const struct biogpt_hparams & hparams,
                struct biogpt_model &   model,
                          ggml_type     wtype,
                          int8_t    verbosity) {

    const int d_model     = hparams.d_model;
    const int n_layer     = hparams.n_layer;
    const int n_positions = hparams.n_positions;

    const int n_mem       = n_layer*n_positions;
    const int n_elements  = n_mem*d_model;

    model.memory_k = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, n_elements);
    model.memory_v = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, n_elements);

    const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

    if (verbosity > 0) {
        printf("%s: memory size = %8.2f MB, n_mem = %d\n", __func__, memory_size/1024.0/1024.0, n_mem);
    }
}

static bool biogpt_model_load(
        const std::string& fname, 
        biogpt_model& model, 
        biogpt_vocab& vocab, 
        uint8_t& verbosity,
        biogpt_progress_callback progress_callback,
        void * progress_callback_user_data) {
    try {
        biogpt_model_load_internal(fname, model, vocab, verbosity, progress_callback, progress_callback_user_data);
        return true;
    } catch (const std::string& err) {
        fprintf(stderr, "%s: error loading model: %s\n", __func__, err.c_str());
        return false;
    }
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
                    ggml_repeat(ctx0, model.layers[layer_ix].attn_layer_norm_w, current), current),
                    ggml_repeat(ctx0, model.layers[layer_ix].attn_layer_norm_b, current)
            );
        }

        // self-attention
        {
            struct ggml_tensor * q_curr = ggml_mul_mat(ctx0, model.layers[layer_ix].q_proj_w, current);
            q_curr = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[layer_ix].q_proj_b, q_curr), q_curr);
            q_curr = ggml_reshape_3d(ctx0, q_curr, d_kv, n_head, N);

            // biogpt scales the query
            q_curr = ggml_scale(ctx0, q_curr, ggml_new_f32(ctx0, 1.0f/sqrt(float(d_kv))));

            struct ggml_tensor * k_curr = ggml_mul_mat(ctx0, model.layers[layer_ix].k_proj_w, current);
            k_curr = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[layer_ix].k_proj_b, k_curr), k_curr);
            k_curr = ggml_reshape_3d(ctx0, k_curr, d_kv, n_head, N);

            struct ggml_tensor * v_curr = ggml_mul_mat(ctx0, model.layers[layer_ix].v_proj_w, current);
            v_curr = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[layer_ix].v_proj_b, v_curr), v_curr);
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
            current = ggml_mul_mat(ctx0, model.layers[layer_ix].o_proj_w, current);
            current = ggml_add(ctx0, current, ggml_repeat(ctx0, model.layers[layer_ix].o_proj_b, current));
        }

        // residual connection
        current = ggml_add(ctx0, current, inpL);

        struct ggml_tensor * inpFF = current;

        // feed forward
        {
            // final layer norm
            current = ggml_norm(ctx0, inpFF);
            current = ggml_add(ctx0, ggml_mul(ctx0, ggml_repeat(ctx0, model.layers[layer_ix].ffn_norm_w, current), current), ggml_repeat(ctx0, model.layers[layer_ix].ffn_norm_b, current));

            // fc1
            current = ggml_mul_mat(ctx0, model.layers[layer_ix].fc_0_w, current);
            current = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[layer_ix].fc_0_b, current), current);

            // gelu
            current = ggml_gelu(ctx0, current);

            // fc2
            current = ggml_mul_mat(ctx0, model.layers[layer_ix].fc_1_w, current);
            current = ggml_add(ctx0, ggml_repeat(ctx0, model.layers[layer_ix].fc_1_b, current), current);
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

        if(!biogpt_model_load(params.model, model, vocab, params.verbosity, )) {
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
