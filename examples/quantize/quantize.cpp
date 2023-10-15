#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "biogpt.h"

static bool biogpt_model_quantize(
        const std::string & fname_inp,
        const std::string & fname_out,
        ggml_ftype ftype) {

    biogpt_model model;

    auto fin = std::ifstream(fname_inp, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    auto fout = std::ofstream(fname_out, std::ios::binary);
    if (!fout) {
        fprintf(stderr, "%s: failed to open '%s' for reading\n", __func__, fname_inp.c_str());
        return false;
    }

    // read and write magic
    {
        uint32_t magic;
        read_safe(fin, magic);
        if (magic != BIOGPT_FILE_MAGIC) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname_inp.c_str());
            return false;
        }

        write_safe(fout, magic);
    }

    auto& hparams = model.hparams;

    // read and write hparams
    {
        read_safe(fin,   hparams.n_vocab);
        read_safe(fin,   hparams.n_layer);
        read_safe(fin,   hparams.n_head);
        read_safe(fin,   hparams.n_positions);
        read_safe(fin,   hparams.d_ff);
        read_safe(fin,   hparams.d_model);
        read_safe(fin,   hparams.ftype);

        write_safe(fout, hparams.n_vocab);
        write_safe(fout, hparams.n_layer);
        write_safe(fout, hparams.n_head);
        write_safe(fout, hparams.n_positions);
        write_safe(fout, hparams.d_ff);
        write_safe(fout, hparams.d_model);
        write_safe(fout, ftype);
    }

    // read and write vocab
    {
        int32_t n_vocab = 0;
        read_safe(fin, n_vocab);
        write_safe(fout, n_vocab);

        if (n_vocab != hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname_inp.c_str(), n_vocab, hparams.n_vocab);
            return false;
        }

        std::string word;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            read_safe (fin,  len);
            write_safe(fout, len);

            if (len > 0) {
                tmp.resize(len);
                fin.read(&tmp[0], tmp.size());
                word.assign(&tmp[0], tmp.size());

                fout.write(&tmp[0], tmp.size());
            }
        }
    }

    // read and write merges
    {
        int32_t n_merges = 0;
        read_safe (fin,  n_merges);
        write_safe(fout, n_merges);

        if (n_merges != hparams.n_merges) {
            fprintf(stderr, "%s: invalid model file '%s' (bad BPE merges size %d != %d)\n",
                    __func__, fname_inp.c_str(), n_merges, hparams.n_merges);
            return false;
        }

        std::string raw_merge;
        std::vector<char> tmp;

        tmp.reserve(128);

        for (int i = 0; i < n_merges; i++) {
            uint32_t len;
            read_safe  (fin, len);
            write_safe(fout, len);

            if (len > 0) {
                tmp.resize(len);
                fin.read(&tmp[0], tmp.size());
                raw_merge.assign(&tmp[0], tmp.size());

                fout.write(&tmp[0], tmp.size());
            }
        }
    }

    try {
        biogpt_model_quantize_internal(fin, fout, ftype);
    } catch(const std::string& err) {
        fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
        return false;
    }

    fin.close();
    fout.close();

    return true;
}

int main(int argc, char **argv) {
    std::string fname_inp, fname_out;
    ggml_ftype ftype;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-f" || arg == "--fname_in") {
            fname_inp = argv[++i];
        } else if (arg == "-o" || arg == "--fname_out") {
            fname_out = argv[++i];
        } else if (arg == "-t" || arg == "--ftype") {
            try {
                ftype = static_cast<ggml_ftype>(std::stoi(argv[++i]));
            } catch (const std::string & err) {
                fprintf(stderr, "error castying file type: %s\n", err.c_str());
            }
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            exit(0);
        }
    }

    biogpt_model_quantize(fname_inp, fname_out, ftype);

    printf("Done.\n");

    return 0;
}
