#include "tokenizer.h"
#include <vector>
#include <numeric>
#include <string>
#include <regex>

using namespace re_patterns;

std::string replace_multidots(std::string& text) {
    std::regex multidot_regex("\\.([\\.]+)");
    std::string replacement("DOTMULTI$1");
    text = std::regex_replace(text, multidot_regex, replacement);

    // Replace DOTMULTI. with DOTDOTMULTI
    std::regex dotmulti_dot_regex("DOTMULTI\\.([^\\.]|\\b)");
    std::string dotdotmulti_replacement("DOTDOTMULTI $1");
    text = std::regex_replace(text, dotmulti_dot_regex, dotdotmulti_replacement);

    // Replace remaining DOTMULTI. with DOTDOTMULTI
    std::regex dotmulti_regex("DOTMULTI\\.");
    std::string dotdotmulti("DOTDOTMULTI");
    text = std::regex_replace(text, dotmulti_regex, dotdotmulti);

    return text;
}

std::string restore_multidots(std::string& text){
    std::string result = text;
    while (std::regex_search(result, std::regex("DOTDOTMULTI"))) {
        result = std::regex_replace(result, std::regex("DOTDOTMULTI"), "DOTMULTI.");
    }
    return std::regex_replace(result, std::regex("DOTMULTI"), ".");
}

std::string escape_xml(std::string& text) {
    std::string result = text;
    for (auto& reg_pat: ESCAPE_XML) {
        std::regex re = reg_pat.first;
        std::string substitution = reg_pat.second;
        result = std::regex_replace(result, re, substitution);
    }
    return result;
}

bool is_lower(const std::string& text) {
    for (char c : text) {
        if (isLower.find(c) == std::string::npos) {
            return false;
        }
    }
    return true;
}

bool is_any_alpha(const std::string& text) {
    for (char c : text) {
        if (isAlpha.find(c) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string handle_nonbreaking_prefixes(const std::string& text, const std::string& lang) {
    // Split the text into tokens to check for non-breaking prefixes.
    std::vector<std::string> tokens;
    std::regex re("\\s+");
    std::sregex_token_iterator iter(text.begin(), text.end(), re, -1);
    std::sregex_token_iterator end;
    for (; iter != end; ++iter) {
        tokens.push_back(*iter);
    }

    std::vector<std::string> nb_prefixes  = nonbreaking_prefixes_words(lang);
    std::vector<std::string> num_prefixes = numeric_only_prefixes(nb_prefixes);

    const int num_tokens = tokens.size();
    for (int i = 0; i < num_tokens; ++i) {
        const std::string& token = tokens[i];
        // Check if token ends with a full stop.
        std::smatch match;
        if (std::regex_search(token, match, std::regex("^(\\S+)\\.$"))) {
            const std::string& prefix = match[1].str();
            // Check for 3 conditions:
            // 1. The prefix contains a full stop and any char in the prefix is within the isAlpha charset.
            // 2. The prefix is in the list of non-breaking prefixes and does not contain #NUMERIC_ONLY#.
            // 3. The token is not the last token and that the next token contains all lowercase.
            if ((prefix.find('.') != std::string::npos && is_any_alpha(prefix))
                || (std::find(nb_prefixes.begin(), nb_prefixes.end(), prefix) != nb_prefixes.end()
                    && std::find(num_prefixes.begin(), num_prefixes.end(), prefix) == num_prefixes.end())
                || (i != num_tokens - 1 && !tokens[i + 1].empty() && is_lower(std::string(tokens[i + 1].front(), 1)))) {
                // No change to the token.
            }
            // Check if the prefix is in NUMERIC_ONLY_PREFIXES and ensures that the next word is a digit.
            else if (std::find(num_prefixes.begin(), num_prefixes.end(), prefix) != num_prefixes.end()
                     && i + 1 < num_tokens && std::regex_search(tokens[i + 1], std::regex("^[0-9]+"))) {
                // No change to the token.
            }
            else {  // Otherwise, add a space after the token before a full stop.
                tokens[i] = prefix + " .";
            }
        }
    }

    // concatenate tokens
    std::string out;
    for (const auto& token : tokens) {
        out += token + " ";
    }
    if (!out.empty()) {
        out.pop_back(); // remove trailing space character
    }
    return out;
}


std::vector<std::string> tokenize(const std::string& text, const std::string& lang) {
    std::string res = text;

    // Deduplicate spaces and clean ASCII junk
    res = std::regex_replace(res, DEDUPLICATE_SPACE, " ");
    res = std::regex_replace(res, ASCII_JUNK, "");

    // strip heading and trailing spaces
    res = std::regex_replace(res, std::regex("^\\s+|\\s+$"), "");

    // Separate special characters outside of IsAlnum character set.
    res = std::regex_replace(res, PAD_NOT_ISALNUM, " $1 ");

    // agressively split dashes
    res = std::regex_replace(res, AGGRESSIVE_HYPHEN_SPLIT, "$1 @-@ ");

    // replace multidots with "DOTDOTMULTI" literal strings.
    replace_multidots(res);

    // separate out "," except if within numbers e.g. 5,300
    for (const auto& pat_reg: COMMA_SEPARATE) {
        std::regex re = pat_reg.first;
        std::string substitution = pat_reg.second;
        res = std::regex_replace(res, re, substitution);
    }

    // language specific apostrophe tokenization
    if (lang  == "en") {
        for (const auto& reg_pat : ENGLISH_SPECIFIC_APOSTROPHE) {
            std::regex re = reg_pat.first;
            std::string substitution = reg_pat.second;
            res = std::regex_replace(res, re, substitution);
        }
    } else if (lang == "fr") {
        for (const auto& reg_pat : FR_IT_SPECIFIC_APOSTROPHE) {
            std::regex re = reg_pat.first;
            std::string substitution = reg_pat.second;
            res = std::regex_replace(res, re, substitution);
        }
    } else {
        res = std::regex_replace(res, NON_SPECIFIC_APOSTROPHE, " ' ");
    }

    // handle non-breaking prefixes
    res = handle_nonbreaking_prefixes(res, lang);

    // clean up extraneous spaces
    res = std::regex_replace(res, DEDUPLICATE_SPACE, " ");
    res = std::regex_replace(res, std::regex("^\\s+|\\s+$"), "");

    // split trailing "."
    res = std::regex_replace(res, TRAILING_DOT_APOSTROPHE, " . ' ");

    // restore multidots
    res = restore_multidots(res);

    // espace XML symbols
    res = escape_xml(res);

    std::vector<std::string> tokens;
    std::stringstream stream(res);
    std::string token;

    while (stream >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

void unit_test(std::string str, std::vector<std::string> expected_tokens) {
    std::vector<std::string> tokens = tokenize(str, "en");

    printf("%s: number of expected tokens = %zu, first 10 tokens: ", __func__, expected_tokens.size());
    for (int i = 0; i < std::min(8, (int) expected_tokens.size()); i++) {
        printf("%s ", expected_tokens[i].c_str());
    }
    printf("\n\n");

    printf("%s: number of actual tokens = %zu, first 10 tokens: ", __func__, tokens.size());
    for (int i = 0; i < std::min(8, (int) tokens.size()); i++) {
        printf("%s ", tokens[i].c_str());
    }
    printf("\n\n");

    assert(tokens.size() == expected_tokens.size());
    for (int i = 0; i < (int) tokens.size(); i++) {
        assert(tokens[i] == expected_tokens[i]);
    }
}

int main() {
    // unit tests
    std::string str1 = "Hello World!";
    std::string str2 = "This ain't funny. It's actually hillarious, yet double Ls. | [] < > [ ] & You're gonna shake it off? Don't?";
    std::string str3 = "this is a webpage https://stackoverflow.com/questions/6181381/how-to-print-variables-in-perl that kicks ass";

    std::vector<std::string> tok1 = {"Hello", "World", "!"};
    std::vector<std::string> tok2 = {"This", "ain", "&apos;t", "funny", ".", "It", "&apos;s", "actually", "hillarious", ",", "yet", "double", "Ls", ".", "&#124;", "&#91;", "&#93;", "&lt;", "&gt;", "&#91;", "&#93;", "&amp;", "You", "&apos;re", "gonna", "shake", "it", "off", "?", "Don", "&apos;t", "?"};
    std::vector<std::string> tok3 = {"this", "is", "a", "webpage", "https", ":", "/", "/", "stackoverflow.com", "/", "questions", "/", "6181381", "/", "how", "@-@", "to", "@-@", "print", "@-@", "variables", "@-@", "in", "@-@", "perl", "that", "kicks", "ass"};

    unit_test(str1, tok1);
    unit_test(str2, tok2);
    unit_test(str3, tok3);

    return 0;
}