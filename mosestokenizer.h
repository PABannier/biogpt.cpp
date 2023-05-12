// Quick implementation extracted from https://github.com/alvations/sacremoses/blob/master/sacremoses/tokenize.py
// This should serve as the first step of the tokenization process for BioGPT.
#pragma once

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>
#include <regex>


std::vector<std::string> moses_tokenize(const std::string& text, const std::string& lang);

std::string moses_detokenize(std::vector<std::string>& in_tokens, const std::string& lang);
