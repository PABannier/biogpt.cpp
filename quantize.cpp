#include "biogpt-util.h"
#include "biogpt.cpp"  // TODO: change to file header

#include <string>
#include <thread>

static bool biogpt_model_quantize(
    const std::string& fname_inp,
    const std::string& fname_out,
    enum biogpt_ftype ftype,
    int nthread) {

    biogpt_file file(fname_out.c_str(), "wb");

    ggml_type quantized_type;
    switch (ftype) {
        case BIOGPT_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_TYPE_Q4_0; break;
        case BIOGPT_FTYPE_MOSTLY_Q8_0: quantized_type = GGML_TYPE_Q8_0; break;
        case BIOGPT_FTYPE_MOSTLY_Q5_0: quantized_type = GGML_TYPE_Q5_0; break;
        default: throw fprintf(stderr, "%s: invalid output file type %d\n", __func__, ftype);
    }

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    biogpt_model model;
    biogpt_vocab vocab;

    if (!biogpt_model_load(fname_inp, model, vocab, 1)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, fname_inp.c_str());
        return 1;
    }

    const auto& hparams = model.hparams;

    // write magic + hparams
    {
        file.write_u32(BIOGPT_FILE_MAGIC);
        file.write_u32(hparams.n_vocab);
        file.write_u32(hparams.n_layer);
        file.write_u32(hparams.n_head);
        file.write_u32(hparams.n_positions);
        file.write_u32(hparams.d_ff);
        file.write_u32(hparams.d_model);
        file.write_u32(quantized_type);
    }

    // write vocab
    {
        uint32_t n_vocab = vocab.n_vocab;
        file.write_u32((uint32_t) n_vocab);

        for (uint32_t i = 0; i < n_vocab; i++) {
            const auto & token = vocab.id_to_token.at(i);
            file.write_u32((uint32_t) token.size());
            file.write_raw(token.data(), token.size());
        }
    }

    // write merges
    {
        uint32_t n_merges = vocab.n_merges;
        file.write_u32((uint32_t) n_merges);

        for (const auto& merge : vocab.bpe_ranks) {
            word_pair pair = merge.first;
            std::string joined_pair = pair.first + " " + pair.second;
            file.write_u32((uint32_t) joined_pair.size());
            file.write_raw(joined_pair.data(), joined_pair.size());
        }
    }

    try {
        biogpt_model_quantize_internal(model, file, quantized_type, nthread);
    } catch(const std::string& err) {
        fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
        return 1;
    }

    file.close();

    return 0;
}

int main(int argc, char **argv) {
    std::string fname_inp;
    std::string fname_out;
    biogpt_ftype ftype;
    int n_thread;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-f" || arg == "--fname_in") {
            fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--fname_out") {
            fname_out = argv[++i];
        } else if (arg == "-t" || arg == "--ftype") {
            try {
                ftype = static_cast<biogpt_ftype>(std::stoi(argv[++i]));
            } catch (const std::string & err) {
                fprintf(stderr, "error castying file type: %s\n", err.c_str());
            }
        } else if (arg == "-n" || arg == "--n_thread") {
            n_thread = std::stoi(argv[++i]);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            exit(0);
        }
    }

    biogpt_model_quantize(fname_inp, fname_out, ftype, n_thread);

    printf("Done.");

    return 0;
}