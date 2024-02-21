#include "mosestokenizer.h"

#include <stdexcept>
#include <unordered_map>
#include <cassert>
#include <vector>
#include <numeric>
#include <string>
#include <regex>

const std::string PERL_UNIPROPS_BASE_PATH = "../data/perluniprops/";
const std::string NONBREAKING_PREFIXES_BASE_PATH = "../data/nonbreaking_prefixes/";

std::vector<std::string> nonbreaking_prefixes_words(std::string lang = "", std::string ignore_lines_startswith = "#") {
    std::vector<std::string> result;

    if (lang.empty()) {
        std::vector<std::string> filenames;
        std::ifstream available_langs_file("data/nonbreaking_prefixes/AVAILABLE_LANGUAGES");
        std::string lang_file;
        while (std::getline(available_langs_file, lang_file)) {
            if (lang_file != "en") {
                filenames.push_back("nonbreaking_prefix." + lang_file);
            }
        }
        filenames.push_back("nonbreaking_prefix.en");

        for (const auto& filename : filenames) {
            std::ifstream file(NONBREAKING_PREFIXES_BASE_PATH + filename);
            std::string line;
            while (std::getline(file, line)) {
                line = line.substr(0, line.find(ignore_lines_startswith));
                if (line.length() > 0) {
                    result.push_back(line);
                }
            }
        }
    } else {
        if (lang != "en") {
            lang = "nonbreaking_prefix." + lang;
        } else {
            lang = "nonbreaking_prefix.en";
        }

        std::ifstream file(NONBREAKING_PREFIXES_BASE_PATH + lang);
        std::string line;
        while (std::getline(file, line)) {
            line = line.substr(0, line.find(ignore_lines_startswith));
            if (line.length() > 0) {
                result.push_back(line);
            }
        }
    }

    // strip heading and trailing spaces
    for(int i = 0; i < (int) result.size(); ++i) {
        result[i] = std::regex_replace(result[i], std::regex("^\\s+|\\s+$"), "");
    }

    return result;
}

bool has_numeric_only(const std::string& text) {
    std::regex regex(R"(\s+#NUMERIC_ONLY#\b)");
    return std::regex_search(text, regex);
}

std::vector<std::string> numeric_only_prefixes(const std::vector<std::string>& prefixes) {
    std::vector<std::string> results;
    for (auto const& w : prefixes) {
        if(has_numeric_only(w)) {
            results.push_back(w.substr(0, w.rfind(" ")));
        }
    }
    return results;
}

std::string perluniprops_chars(const std::string & category) {
    std::string fpath = PERL_UNIPROPS_BASE_PATH + category + ".txt";
    std::ifstream fin(fpath, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Perl Uniprops file not available.");
    }
    std::stringstream buffer;
    buffer << fin.rdbuf();
    std::string contents = buffer.str();

    // std::vector<std::string> chars;
    std::string res;
    for (char ch : contents) {
        // chars.push_back(std::string(1, ch));
        res += ch;
    }

    return res;
    // return chars;
}

namespace re_patterns {
    std::string isAlnum = perluniprops_chars("IsAlnum");
    std::string isAlpha = perluniprops_chars("IsAlpha");
    std::string isLower = perluniprops_chars("IsLower");
    std::string isN     = perluniprops_chars("IsN");
    std::string isSc    = perluniprops_chars("IsSc");

    std::regex DEDUPLICATE_SPACE("\\s+");
    std::regex ASCII_JUNK       ("[\\x00-\\x1F]");

    std::regex AGGRESSIVE_HYPHEN_SPLIT("(["+ isAlnum +"])\\-(?=[" + isAlnum + "])");
    std::regex AGGRESSIVE_HYPHEN_SPLIT_DETOK(" @-@");

    std::regex PAD_NOT_ISALNUM("([^" + isAlnum + "\\s\\.'\\`\\,\\-])");

    std::regex ONE_SPACE(" {2,}");

    std::vector<std::pair<std::regex, std::string>> COMMA_SEPARATE = {
        { std::regex("([^" + isN + "])[,]"), "$1 , " },
        { std::regex("[,]([^" + isN + "])"), " , $1" },
        { std::regex("([" + isN + "])[,]$"), "$1 , " }
    };

    std::vector<std::pair<std::regex, std::string>> ENGLISH_SPECIFIC_APOSTROPHE = {
        { std::regex("([^" + isAlpha + "])[']([^" + isAlpha + "])")      , "$1 ' $2" },
        { std::regex("([^" + isAlpha + isN + "])[']([" + isAlpha + "])") , "$1 ' $2" },
        { std::regex("([" + isAlpha + "])[']([^" + isAlpha + "])")       , "$1 ' $2" },
        { std::regex("([" + isAlpha + "])[']([" + isAlpha + "])")        , "$1 '$2"  },
        { std::regex("([" + isN + "])[']([s])")                          , "$1 '$2"  }
    };

    std::vector<std::pair<std::regex, std::string>> FR_IT_SPECIFIC_APOSTROPHE = {
        { std::regex{"([^" + isAlpha + "])[']([^"  + isAlpha + "])"}, "$1 ' $2" },
        { std::regex{"([^" + isAlpha + "])['](["   + isAlpha + "])"}, "$1 ' $2" },
        { std::regex{"(["  + isAlpha  + "])[']([^" + isAlpha + "])"}, "$1 ' $2" },
        { std::regex{"(["  + isAlpha  + "])['](["  + isAlpha + "])"}, "$1' $2"  }
    };

    std::vector<std::pair<std::regex, std::string>> ESCAPE_XML = {
        { std::regex("&")  , "&amp;"  },
        { std::regex("\\|"), "&#124;" },
        { std::regex("<")  , "&lt;"   },
        { std::regex(">")  , "&gt;"   },
        { std::regex("\'") , "&apos;" },
        { std::regex("\"") , "&quot;" },
        { std::regex("\\["), "&#91;"  },
        { std::regex("\\]")  , "&#93;"  }
    };

    std::pair<std::regex, std::string> UNESCAPE_FACTOR_SEPARATOR("&#124;", "|");
    std::pair<std::regex, std::string> UNESCAPE_LEFT_ANGLE_BRACKET("&lt;", "<");
    std::pair<std::regex, std::string> UNESCAPE_RIGHT_ANGLE_BRACKET("&gt;", ">");
    std::pair<std::regex, std::string> UNESCAPE_DOUBLE_QUOTE("&quot;", "\"");
    std::pair<std::regex, std::string> UNESCAPE_SINGLE_QUOTE("&apos;", "'");
    std::pair<std::regex, std::string> UNESCAPE_SYNTAX_NONTERMINAL_LEFT("&#91;", "[");
    std::pair<std::regex, std::string> UNESCAPE_SYNTAX_NONTERMINAL_RIGHT("&#93;", "]");
    std::pair<std::regex, std::string> UNESCAPE_AMPERSAND("&amp;", "&");

    // The legacy regexes are used to support outputs from older Moses versions.
    std::pair<std::regex, std::string> UNESCAPE_FACTOR_SEPARATOR_LEGACY("&bar;", "|");
    std::pair<std::regex, std::string> UNESCAPE_SYNTAX_NONTERMINAL_LEFT_LEGACY("&bra;", "[");
    std::pair<std::regex, std::string> UNESCAPE_SYNTAX_NONTERMINAL_RIGHT_LEGACY("&ket;", "]");

    std::vector<std::pair<std::regex, std::string>> UNESCAPE_XML = {
        UNESCAPE_FACTOR_SEPARATOR_LEGACY,
        UNESCAPE_FACTOR_SEPARATOR,
        UNESCAPE_LEFT_ANGLE_BRACKET,
        UNESCAPE_RIGHT_ANGLE_BRACKET,
        UNESCAPE_SYNTAX_NONTERMINAL_LEFT_LEGACY,
        UNESCAPE_SYNTAX_NONTERMINAL_RIGHT_LEGACY,
        UNESCAPE_DOUBLE_QUOTE,
        UNESCAPE_SINGLE_QUOTE,
        UNESCAPE_SYNTAX_NONTERMINAL_LEFT,
        UNESCAPE_SYNTAX_NONTERMINAL_RIGHT,
        UNESCAPE_AMPERSAND
    };

    std::regex NON_SPECIFIC_APOSTROPHE("\'");

    std::regex TRAILING_DOT_APOSTROPHE("\\.' ?$");
}

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


std::vector<std::string> moses_tokenize(const std::string& text, const std::string& lang) {
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

std::string moses_detokenize(std::vector<std::string>& in_tokens, const std::string& lang) {
    std::string text;
    std::vector<std::string> tokens;

    // Convert the list of tokens into a string and pad it with spaces.
    {
        text += " ";
        for (auto& token : in_tokens) {
            text += token + " ";
        }
    }

    // Detokenize the agressive hyphen split.
    text = std::regex_replace(text, AGGRESSIVE_HYPHEN_SPLIT_DETOK, "-");

    // Unescape XML symbols.
    for (auto& pair : UNESCAPE_XML) {
        std::regex reg_pat = pair.first;
        std::string substitution = pair.second;
        std::regex_replace(text, reg_pat, substitution);
    }

    std::unordered_map<std::string, int> quote_counts = {{"'", 0}, {"\"", 0}, {"``", 0}, {"`", 0}, {"''", 0}};

    std::string prepend_space = " ";
    std::string detokenized_text = "";

    // Split text into tokens.
    {
        std::stringstream stream(text);
        std::string token;

        while (stream >> token) {
            tokens.push_back(token);
        }
    }

    for (int i = 0; i < (int) tokens.size(); i++) {
        std::string token = tokens[i];

        if (std::regex_search(token, std::regex("^[" + isSc + "\\(\\[\\{\\¿\\¡]+$"))) {
            detokenized_text += prepend_space + token;
            prepend_space = "";
        }

        else if (std::regex_search(token, std::regex("^\\[\\,\\.\\?\\!\\:\\;\\\\\\%\\}\\]\\)+$"))) {
            if (lang == "fr" && std::regex_search(token, std::regex("^\\[\\?\\!\\:\\;\\\\\\%\\]+$"))) {
                detokenized_text += " ";
            }
            detokenized_text += token;
            prepend_space = " ";
        }

        else if (lang == "en" && i > 0 && std::regex_search(token, std::regex("^['][" + isAlpha + "]"))) {
            detokenized_text += token;
            prepend_space = " ";
        }

        else if (lang == "fr" || lang == "it" || lang == "ga") {
            if (i <= (int) tokens.size() - 2 && std::regex_search(token, std::regex("[" + isAlpha + "][']$")) && std::regex_search(tokens[i + 1], std::regex("^[" + isAlpha + "]"))) {
                detokenized_text += prepend_space + token;
                prepend_space = "";
            }
        }

        else if (std::regex_search(token, std::regex("^['\"„“`]+$"))) {
            std::string normalized_quo = token;

            if (std::regex_search(token, std::regex("^[„“”]+$"))) {
                normalized_quo = "\"";
            }

            if (quote_counts.find(normalized_quo) == quote_counts.end()) {
                quote_counts[normalized_quo] = 0;
            }

            if (quote_counts[normalized_quo] % 2 == 0) {
                if (lang == "en" && token == "'" && i > 0 && std::regex_search(tokens[i - 1], std::regex("[s]$"))) {
                    detokenized_text += token;
                    prepend_space = " ";
                }
                else {
                    detokenized_text += prepend_space + token;
                    prepend_space = "";
                    quote_counts[normalized_quo] += 1;
                }
            }
            else {
                detokenized_text += token;
                prepend_space = " ";
                quote_counts[normalized_quo] += 1;
            }
        }

        else {
            detokenized_text += prepend_space + token;
            prepend_space = " ";
        }
    }

    detokenized_text = std::regex_replace(detokenized_text, ONE_SPACE, " ");

    // Remove heading and trailing spaces
    detokenized_text = std::regex_replace(detokenized_text, std::regex("^\\s+|\\s+$"), "");

    return detokenized_text;
}

void unit_test(std::string str, std::vector<std::string> expected_tokens) {
    std::vector<std::string> tokens = moses_tokenize(str, "en");

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

int run_unit_tests() {
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
