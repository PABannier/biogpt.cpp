// Quick reimplementation of byte-pair encoding from
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/biogpt/tokenization_biogpt.py
#pragma once

#include <map>
#include <string>

typedef std::pair<std::string, std::string> word_pair;

std::string bpe(const std::string& token, std::map<word_pair, int>& bpe_ranks);
