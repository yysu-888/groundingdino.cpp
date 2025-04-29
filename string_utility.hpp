#pragma once
#include "fstream"
#include "iostream"
#include "map"
#include "string"
#include "vector"

#define LEFTSTRIP 0
#define RIGHTSTRIP 1
#define BOTHSTRIP 2

std::string do_strip(const std::string& str, int striptype, const std::string& chars) {
    std::string::size_type strlen   = str.size();
    std::string::size_type charslen = chars.size();
    std::string::size_type i, j;

    if (0 == charslen) {
        i = 0;
        if (striptype != RIGHTSTRIP) {
            while (i < strlen && ::isspace(str[i])) {
                i++;
            }
        }
        j = strlen;
        if (striptype != LEFTSTRIP) {
            j--;
            while (j >= i && ::isspace(str[j])) {
                j--;
            }
            j++;
        }
    } else {
        const char* sep = chars.c_str();
        i               = 0;
        if (striptype != RIGHTSTRIP) {
            while (i < strlen && memchr(sep, str[i], charslen)) {
                i++;
            }
        }
        j = strlen;
        if (striptype != LEFTSTRIP) {
            j--;
            while (j >= i && memchr(sep, str[j], charslen)) {
                j--;
            }
            j++;
        }
        if (0 == i && j == strlen) {
            return str;
        } else {
            return str.substr(i, j - i);
        }
    }
}

std::string strip(const std::string& str, const std::string& chars = " ") {
    return do_strip(str, BOTHSTRIP, chars);
}

std::string lstrip(const std::string& str, const std::string& chars = " ") {
    return do_strip(str, LEFTSTRIP, chars);
}

std::string rstrip(const std::string& str, const std::string& chars = " ") {
    return do_strip(str, RIGHTSTRIP, chars);
}

int startswith(std::string s, std::string sub) {
    return s.find(sub) == 0 ? 1 : 0;
}

int endswith(std::string s, std::string sub) {
    return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
}
