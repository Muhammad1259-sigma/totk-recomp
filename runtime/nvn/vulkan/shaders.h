// Shader compilation and reflection for the Vulkan backend (shaders.c).
#pragma once
#include <stdbool.h>
#include <stddef.h>

#include "vk.h"

typedef struct { uint32_t set, binding; VkDescriptorType type; } VkBindingInfo;

void vk_shaders_init(void);
uint32_t *vk_glsl_to_spirv(const char *src, int stage, const char *name, size_t *nwords);  // malloc'd, NULL on error
bool vk_spirv_cached(const char *src, int stage);  // src's SPIR-V is in the disk cache
uint64_t vk_spirv_key(const char *src, int stage);  // its disk-cache key
uint32_t *vk_spirv_load(uint64_t key, size_t *nwords);  // cached SPIR-V by key (malloc'd), NULL if missing
VkShaderModule vk_module(const uint32_t *spv, size_t nwords);
// Descriptor bindings declared by a SPIR-V module.
int vk_reflect(const uint32_t *spv, size_t nwords, VkBindingInfo *out, int max);
VkDescriptorSetLayout vk_set_layout(const VkBindingInfo *b, int n, uint32_t set, VkShaderStageFlags stages);
