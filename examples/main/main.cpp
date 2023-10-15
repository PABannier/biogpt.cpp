#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"

#include "biogpt.h"


int main(int argc, char **argv) {
    ggml_time_init();

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

        if(!biogpt_model_load(params.model, model, vocab, params.verbosity)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // keep this buffer alive while evaluating the model
    ggml_backend_buffer_t buf_compute;

    struct ggml_allocr * allocr = NULL;
    // allocate the compute buffer
    {
         // alignment required by the backend
        size_t align = ggml_backend_get_alignment(model.backend);
        allocr = ggml_allocr_new_measure(align);

        // create the worst case graph for memory usage estimation
        int n_tokens = std::min(model.hparams.n_positions, params.n_batch);
        int n_past = model.hparams.n_positions - n_tokens;
        struct ggml_cgraph * gf = biogpt_graph(model, allocr, token_sequence(n_tokens, 0), n_past);

        // compute the required memory
        size_t mem_size = ggml_allocr_alloc_graph(allocr, gf);

        // recreate the allocator with the required memory
        ggml_allocr_free(allocr);
        buf_compute = ggml_backend_alloc_buffer(model.backend, mem_size);
        allocr = ggml_allocr_new_from_buffer(buf_compute);

        fprintf(stderr, "%s: compute buffer size: %.2f MB\n", __func__, mem_size/1024.0/1024.0);
    }

    int n_past = 0;

    int64_t t_sample_us  = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    token_sequence embed_inp = gpt_tokenize(vocab, params.prompt, params.lang);

    params.n_predict = std::min(params.n_predict, model.hparams.n_positions - (int) embed_inp.size());

    printf("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
    printf("%s: number of tokens in prompt = %zu, first 8 tokens: ", __func__, embed_inp.size());
    for (int i = 0; i < std::min(8, (int) embed_inp.size()); i++) {
        printf("%d ", embed_inp[i]);
    }
    printf("\n\n");

    token_sequence embed;

    for (size_t i = embed.size(); i < (int) embed_inp.size() + params.n_predict; i++) {
        // predict
        if (embed.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if(!biogpt_eval(model, embed, logits, allocr, n_past, params.n_threads)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embed.size();
        embed.clear();

        if (i >= embed_inp.size()) {
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
            for (size_t k = i; k < embed_inp.size(); k++) {
                embed.push_back(embed_inp[k]);
                if (int32_t(embed.size()) >= params.n_batch) {
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
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    ggml_free(model.ctx);

    ggml_backend_buffer_free(model.buffer_w);
    ggml_backend_buffer_free(model.buffer_kv);
    ggml_backend_buffer_free(buf_compute);
    ggml_backend_free(model.backend);

    return 0;
}