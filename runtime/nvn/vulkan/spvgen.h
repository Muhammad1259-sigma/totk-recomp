// spvgen: direct GLSL -> SPIR-V 1.6 for the TotK translator's GLSL subset (same function body as glslang 15.1 / shaderc).
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Returns a malloc'd SPIR-V module (free() it) and its size in words, or NULL when the shader is outside the subset
// (*why = the reason, a static string); the caller then compiles it with shaderc. Thread-safe (state is thread-local).
uint32_t *spvgen(const char *src, size_t len, int fs, size_t *nwords, const char **why);
#ifdef __cplusplus
}
#endif
