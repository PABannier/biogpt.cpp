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
            std::ifstream file("data/nonbreaking_prefixes/" + filename);
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

        std::ifstream file("data/nonbreaking_prefixes/" + lang);
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


std::string perluniprops_chars(const std::string& category) {
    std::string fpath = "data/perluniprops/" + category + ".txt";
    std::ifstream fin(fpath, std::ios::binary);
    if (!fin) {
        throw "Perl Uniprops file not available";
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
    std::string isAlnum = perluniprops_chars("isAlnum");
    std::string isAlpha = perluniprops_chars("isAlpha");
    std::string isLower = perluniprops_chars("isLower");
    std::string isN     = perluniprops_chars("isN");

    std::regex DEDUPLICATE_SPACE("\\s+");
    std::regex ASCII_JUNK       ("[\\x00-\\x1F]");

    std::regex AGGRESSIVE_HYPHEN_SPLIT("(["+ isAlnum +"])\\-(?=[" + isAlnum + "])");

    std::regex PAD_NOT_ISALNUM("([^" + isAlnum + "\\s\\.'\\`\\,\\-])");

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

    std::regex NON_SPECIFIC_APOSTROPHE("\'");

    std::regex TRAILING_DOT_APOSTROPHE("\\.' ?$");
}


std::vector<std::string> tokenize(const std::string& text, const std::string& lang);
