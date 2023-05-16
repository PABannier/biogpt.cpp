// Quick reimplementation of byte-pair encoding from
// https://github.com/huggingface/transformers/blob/main/src/transformers/models/biogpt/tokenization_biogpt.py
#pragma once

#include "biogpt-util.h"

#include <map>
#include <string>

std::string bpe(const std::string& token, std::map<word_pair, int>& bpe_ranks);
