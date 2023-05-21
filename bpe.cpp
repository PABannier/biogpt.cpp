#include "bpe.h"
#include <algorithm>
#include <iostream>
#include <set>
#include <map>
#include <vector>

std::set<word_pair> get_pairs(const std::vector<std::string>& subwords) {
    std::set<word_pair> pairs;
    std::string prev_subword = subwords[0];
    for (int i = 1; i < (int) subwords.size(); i++) {
        std::string subword = subwords[i];
        word_pair pair(prev_subword, subword);
        pairs.insert(pair);
        prev_subword = subword;
    }
    return pairs;
}

std::string bpe(const std::string& token, std::map<word_pair, int>& bpe_ranks) {
    std::vector<std::string> word;
    for (int i = 0; i < (int) token.size() - 1; i++) {
        word.push_back(std::string(1, token[i]));
    }
    word.push_back(token.substr(token.size() - 1) + "</w>");

    std::set<word_pair> pairs = get_pairs(word);

    if (pairs.empty()) {
        return token + "</w>";
    }

    while (true) {
        auto it = std::min_element(pairs.begin(), pairs.end(), [&](const word_pair& a, const word_pair& b) {
            if (bpe_ranks.find(a) == bpe_ranks.end()) {
                return false;
            } else if (bpe_ranks.find(b) == bpe_ranks.end()) {
                return true;
            }
            return bpe_ranks.at(a) < bpe_ranks.at(b);
        });

        word_pair bigram = *it;

        if (bpe_ranks.find(bigram) == bpe_ranks.end()) {
            break;
        }

        std::string first = bigram.first;
        std::string second = bigram.second;
        std::vector<std::string> new_word;
        int i = 0;

        while (i < (int) word.size()) {
            auto it = std::find(word.begin() + i, word.end(), first);
            if (it == word.end()) {
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }
            new_word.insert(new_word.end(), word.begin() + i, it);
            i = std::distance(word.begin(), it);

            if (word[i] == first && i < (int) word.size() - 1 && word[i + 1] == second) {
                new_word.push_back(first + second);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i += 1;
            }
        }

        word = new_word;

        if(word.size() == 1) {
            break;
        }
        pairs = get_pairs(word);
    }

    std::string result;
    for (const auto& w : word) {
        result += w + " ";
    }
    result = result.substr(0, result.size() - 1);  // remove last " "

    if (result == "\n  </w>") {
        result = "\n</w>";
    }

    return result;
}
