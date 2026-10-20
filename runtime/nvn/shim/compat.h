// TotK: POSIX bits the LASRecomp backend uses, for the clang/UCRT (MSVC ABI) build. Force-included (-include).
#pragma once
#include <direct.h>
#include <io.h>
#include <string.h>
#include <sys/stat.h>
#define mkdir(path, mode) _mkdir(path)
static inline size_t strlcpy(char *d, const char *s, size_t n) {
    size_t l = strlen(s);
    if (n) { size_t k = l < n - 1 ? l : n - 1; memcpy(d, s, k); d[k] = 0; }
    return l;
}
