// Vulkan textures: NVN formats, images, change-detected uploads from guest memory, views and samplers
// (counterpart of the texture half of metal.m and the view helpers of renderer.m).
#include <SDL3/SDL_timer.h>
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#include "vk.h"
#include "../astc.h"
#include "../shim/las.h"  // TotK: was ../renderer.h
#include "../shim/las.h"  // TotK: was ../texpatch.h
extern int vk_up_kind;   // (exec.c upload stats by kind)

// ================================================================== formats
typedef struct { VkFormat vf; uint8_t expand_rgb, alpha_one; } VkFmt;

static VkFmt fmt_of(uint32_t f) {
    switch (f) {
    case 0x01: return (VkFmt){VK_FORMAT_R8_UNORM};
    case 0x02: return (VkFmt){VK_FORMAT_R8_SNORM};
    case 0x03: return (VkFmt){VK_FORMAT_R8_UINT};
    case 0x04: return (VkFmt){VK_FORMAT_R8_SINT};
    case 0x05: return (VkFmt){VK_FORMAT_R16_SFLOAT};
    case 0x06: return (VkFmt){VK_FORMAT_R16_UNORM};
    case 0x07: return (VkFmt){VK_FORMAT_R16_SNORM};
    case 0x08: return (VkFmt){VK_FORMAT_R16_UINT};
    case 0x09: return (VkFmt){VK_FORMAT_R16_SINT};
    case 0x0A: return (VkFmt){VK_FORMAT_R32_SFLOAT};
    case 0x0B: return (VkFmt){VK_FORMAT_R32_UINT};
    case 0x0C: return (VkFmt){VK_FORMAT_R32_SINT};
    case 0x0D: return (VkFmt){VK_FORMAT_R8G8_UNORM};
    case 0x0E: return (VkFmt){VK_FORMAT_R8G8_SNORM};
    case 0x0F: return (VkFmt){VK_FORMAT_R8G8_UINT};
    case 0x10: return (VkFmt){VK_FORMAT_R8G8_SINT};
    case 0x11: return (VkFmt){VK_FORMAT_R16G16_SFLOAT};
    case 0x12: return (VkFmt){VK_FORMAT_R16G16_UNORM};
    case 0x13: return (VkFmt){VK_FORMAT_R16G16_SNORM};
    case 0x14: return (VkFmt){VK_FORMAT_R16G16_UINT};
    case 0x15: return (VkFmt){VK_FORMAT_R16G16_SINT};
    case 0x16: return (VkFmt){VK_FORMAT_R32G32_SFLOAT};
    case 0x17: return (VkFmt){VK_FORMAT_R32G32_UINT};
    case 0x18: return (VkFmt){VK_FORMAT_R32G32_SINT};
    case 0x19: return (VkFmt){VK_FORMAT_R8G8B8A8_UNORM, 1, 1};  // RGB8: expanded on upload
    case 0x25: return (VkFmt){VK_FORMAT_R8G8B8A8_UNORM};
    case 0x26: return (VkFmt){VK_FORMAT_R8G8B8A8_SNORM};
    case 0x27: return (VkFmt){VK_FORMAT_R8G8B8A8_UINT};
    case 0x28: return (VkFmt){VK_FORMAT_R8G8B8A8_SINT};
    case 0x29: return (VkFmt){VK_FORMAT_R16G16B16A16_SFLOAT};
    case 0x2A: return (VkFmt){VK_FORMAT_R16G16B16A16_UNORM};
    case 0x2B: return (VkFmt){VK_FORMAT_R16G16B16A16_SNORM};
    case 0x2C: return (VkFmt){VK_FORMAT_R16G16B16A16_UINT};
    case 0x2D: return (VkFmt){VK_FORMAT_R16G16B16A16_SINT};
    case 0x2E: return (VkFmt){VK_FORMAT_R32G32B32A32_SFLOAT};
    case 0x2F: return (VkFmt){VK_FORMAT_R32G32B32A32_UINT};
    case 0x30: return (VkFmt){VK_FORMAT_R32G32B32A32_SINT};
    case 0x31: return (VkFmt){VK_FORMAT_S8_UINT};
    case 0x32: {   // LARECOMP_D16_AS_D32=1: A/B for depth-EQUAL material passes (TotK lighting quads vs a D16 class buffer)
        static int d32 = -1;
        if (d32 < 0) d32 = getenv("LARECOMP_D16_AS_D32") != NULL;
        return (VkFmt){d32 ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_D16_UNORM};
    }
    case 0x33: return (VkFmt){VK_FORMAT_D32_SFLOAT};  // DEPTH24 (as on Metal: no 24-bit depth there)
    case 0x34: return (VkFmt){VK_FORMAT_D32_SFLOAT};
    case 0x35: return (VkFmt){VK_FORMAT_D32_SFLOAT_S8_UINT};
    case 0x36: return (VkFmt){VK_FORMAT_D32_SFLOAT_S8_UINT};
    case 0x37: return (VkFmt){VK_FORMAT_R8G8B8A8_SRGB, 0, 1};
    case 0x38: return (VkFmt){VK_FORMAT_R8G8B8A8_SRGB};
    case 0x39: return (VkFmt){VK_FORMAT_R4G4B4A4_UNORM_PACK16};   // Metal ABGR4Unorm
    case 0x3A: return (VkFmt){VK_FORMAT_R5G6B5_UNORM_PACK16};     // Metal B5G6R5Unorm
    case 0x3B: return (VkFmt){VK_FORMAT_R5G5B5A1_UNORM_PACK16};   // Metal A1BGR5Unorm
    case 0x3C: return (VkFmt){VK_FORMAT_R5G6B5_UNORM_PACK16};
    case 0x3D: return (VkFmt){VK_FORMAT_A2B10G10R10_UNORM_PACK32};
    case 0x3E: return (VkFmt){VK_FORMAT_A2B10G10R10_UINT_PACK32};
    case 0x3F: return (VkFmt){VK_FORMAT_B10G11R11_UFLOAT_PACK32};
    case 0x40: return (VkFmt){VK_FORMAT_E5B9G9R9_UFLOAT_PACK32};
    case 0x41: case 0x42: return (VkFmt){VK_FORMAT_BC1_RGBA_UNORM_BLOCK};
    case 0x43: return (VkFmt){VK_FORMAT_BC2_UNORM_BLOCK};
    case 0x44: return (VkFmt){VK_FORMAT_BC3_UNORM_BLOCK};
    case 0x45: case 0x46: return (VkFmt){VK_FORMAT_BC1_RGBA_SRGB_BLOCK};
    case 0x47: return (VkFmt){VK_FORMAT_BC2_SRGB_BLOCK};
    case 0x48: return (VkFmt){VK_FORMAT_BC3_SRGB_BLOCK};
    case 0x49: return (VkFmt){VK_FORMAT_BC4_UNORM_BLOCK};
    case 0x4A: return (VkFmt){VK_FORMAT_BC4_SNORM_BLOCK};
    case 0x4B: return (VkFmt){VK_FORMAT_BC5_UNORM_BLOCK};
    // 0x4C = signed RGTC2; LARECOMP_BC5_UNORM=1 decodes it unsigned (see metal.m / notes/PLAN.md)
    case 0x4C: return (VkFmt){getenv("LARECOMP_BC5_UNORM") ? VK_FORMAT_BC5_UNORM_BLOCK : VK_FORMAT_BC5_SNORM_BLOCK};
    case 0x4D: return (VkFmt){VK_FORMAT_BC7_UNORM_BLOCK};
    case 0x4E: return (VkFmt){VK_FORMAT_BC7_SRGB_BLOCK};
    case 0x4F: return (VkFmt){VK_FORMAT_BC6H_SFLOAT_BLOCK};
    case 0x50: return (VkFmt){VK_FORMAT_BC6H_UFLOAT_BLOCK};
    case 0x6D: return (VkFmt){VK_FORMAT_R8G8B8A8_UNORM, 0, 1};
    case 0x71: return (VkFmt){VK_FORMAT_R16G16B16A16_SFLOAT, 0, 1};
    case 0x76: return (VkFmt){VK_FORMAT_R32G32B32A32_SFLOAT, 0, 1};
    case 0x79: return (VkFmt){VK_FORMAT_ASTC_4x4_UNORM_BLOCK};
    case 0x7A: return (VkFmt){VK_FORMAT_ASTC_5x4_UNORM_BLOCK};
    case 0x7B: return (VkFmt){VK_FORMAT_ASTC_5x5_UNORM_BLOCK};
    case 0x7C: return (VkFmt){VK_FORMAT_ASTC_6x5_UNORM_BLOCK};
    case 0x7D: return (VkFmt){VK_FORMAT_ASTC_6x6_UNORM_BLOCK};
    case 0x7E: return (VkFmt){VK_FORMAT_ASTC_8x5_UNORM_BLOCK};
    case 0x7F: return (VkFmt){VK_FORMAT_ASTC_8x6_UNORM_BLOCK};
    case 0x80: return (VkFmt){VK_FORMAT_ASTC_8x8_UNORM_BLOCK};
    case 0x81: return (VkFmt){VK_FORMAT_ASTC_10x5_UNORM_BLOCK};
    case 0x82: return (VkFmt){VK_FORMAT_ASTC_10x6_UNORM_BLOCK};
    case 0x83: return (VkFmt){VK_FORMAT_ASTC_10x8_UNORM_BLOCK};
    case 0x84: return (VkFmt){VK_FORMAT_ASTC_10x10_UNORM_BLOCK};
    case 0x85: return (VkFmt){VK_FORMAT_ASTC_12x10_UNORM_BLOCK};
    case 0x86: return (VkFmt){VK_FORMAT_ASTC_12x12_UNORM_BLOCK};
    case 0x87: return (VkFmt){VK_FORMAT_ASTC_4x4_SRGB_BLOCK};
    case 0x88: return (VkFmt){VK_FORMAT_ASTC_5x4_SRGB_BLOCK};
    case 0x89: return (VkFmt){VK_FORMAT_ASTC_5x5_SRGB_BLOCK};
    case 0x8A: return (VkFmt){VK_FORMAT_ASTC_6x5_SRGB_BLOCK};
    case 0x8B: return (VkFmt){VK_FORMAT_ASTC_6x6_SRGB_BLOCK};
    case 0x8C: return (VkFmt){VK_FORMAT_ASTC_8x5_SRGB_BLOCK};
    case 0x8D: return (VkFmt){VK_FORMAT_ASTC_8x6_SRGB_BLOCK};
    case 0x8E: return (VkFmt){VK_FORMAT_ASTC_8x8_SRGB_BLOCK};
    case 0x8F: return (VkFmt){VK_FORMAT_ASTC_10x5_SRGB_BLOCK};
    case 0x90: return (VkFmt){VK_FORMAT_ASTC_10x6_SRGB_BLOCK};
    case 0x91: return (VkFmt){VK_FORMAT_ASTC_10x8_SRGB_BLOCK};
    case 0x92: return (VkFmt){VK_FORMAT_ASTC_10x10_SRGB_BLOCK};
    case 0x93: return (VkFmt){VK_FORMAT_ASTC_12x10_SRGB_BLOCK};
    case 0x94: return (VkFmt){VK_FORMAT_ASTC_12x12_SRGB_BLOCK};
    case 0x95: case 0x96: case 0x97: return (VkFmt){VK_FORMAT_R5G5B5A1_UNORM_PACK16};
    case 0x98: return (VkFmt){VK_FORMAT_B8G8R8A8_UNORM, 0, 1};
    case 0x99: return (VkFmt){VK_FORMAT_B8G8R8A8_UNORM};
    case 0x9A: return (VkFmt){VK_FORMAT_B8G8R8A8_SRGB, 0, 1};
    case 0x9B: return (VkFmt){VK_FORMAT_B8G8R8A8_SRGB};
    default: return (VkFmt){VK_FORMAT_R8G8B8A8_UNORM};
    }
}

bool renderer_format_is_depth(uint32_t f) { return f >= 0x31 && f <= 0x36; }
bool renderer_format_has_stencil(uint32_t f) { return f == 0x31 || f == 0x35 || f == 0x36; }
VkFormat vk_format(uint32_t f) { return fmt_of(f).vf; }
bool vk_format_alpha_one(uint32_t f) { return fmt_of(f).alpha_one; }
bool renderer_format_alpha_one(uint32_t f) { return fmt_of(f).alpha_one; }

static VkImageAspectFlags aspect_of(uint32_t f) {
    if (f == 0x31) return VK_IMAGE_ASPECT_STENCIL_BIT;
    if (f == 0x35 || f == 0x36) return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    if (renderer_format_is_depth(f)) return VK_IMAGE_ASPECT_DEPTH_BIT;
    return VK_IMAGE_ASPECT_COLOR_BIT;
}

static bool format_supports(VkFormat f, VkFormatFeatureFlags want) {
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(g_vk.phys, f, &p);
    return (p.optimalTilingFeatures & want) == want;
}

// ================================================================== output resolution / per-frame clears
// Same rules as metal.m (see the comments there).
static float g_rs_x = 1, g_rs_y = 1;
void renderer_get_scale(float *sx, float *sy) { *sx = g_rs_x; *sy = g_rs_y; }
void renderer_set_scale(float sx, float sy) { g_rs_x = sx; g_rs_y = sy; }
uint64_t renderer_present_count(void);

// Output canvas (TOTK_OUT_SCALE, 2026-10-17): TotK's 1920x1080 targets (the canvas its upscale pass writes and the UI
// draws on, the window textures, the menus' layers and 3D model buffers) at the window's resolution (device.c: viewport height /
// 1080), so the scene reaches the window in one resample (exec.c, the upscale pass) and the UI draws at full size.
static float g_out_s = 1;
static int out_scale_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("TOTK_OUT_SCALE"); on = e && *e && strcmp(e, "0") != 0; }
    return on;
}
void renderer_set_out_scale(float s) { if (out_scale_on()) g_out_s = s; }

// Shadow maps: square depth render targets (TotK: the 800x800 x4 sun cascades, D32) rendered at TOTK_SHADOW_SCALE x
// their size (default 2; 1 = the game's own). The user (2026-10-15): "theres still alot of shimmering on the shadows".
// Shaders see the guest size through the rscale table like every scaled target (translate.c).
static float shadow_scale(void) {
    static float s = -1;
    if (s < 0) {
        const char *e = getenv("TOTK_SHADOW_SCALE");
        s = e && *e ? (float)atof(e) : 2.0f;
        if (!(s >= 1.0f && s <= 4.0f)) s = 1.0f;
    }
    return s;
}
// Promoted targets (U21, 2026-10-17): TotK renders some non-compressible targets in one pass with a scaled one of the
// same size (fs dd9edc34: an R8 1600x900 mask with the scene's depth buffer, every frame; fs 7e405a3d: R11G11B10F + R8
// 800x450 while loading). begin_pass renders the smallest attachment extent, so at TOTK_RES_SCALE=2 the pass used a
// quarter of the scaled attachment. exec.c begin_pass promotes the unscaled one (storage address + size) to the scene's
// scale from its next backend (TotK re-creates its targets every frame). LARECOMP_NO_PROMOTE=1: off.
#define PROMO_MAX 64
static struct { uint64_t addr; uint32_t w, h; } g_promo[PROMO_MAX];
static int g_npromo;
static bool tex_promoted(const NvnTexture *t) {
    int n = __atomic_load_n(&g_npromo, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++)
        if (g_promo[i].addr == t->addr && g_promo[i].w == t->width && g_promo[i].h == t->height) return true;
    return false;
}
void renderer_texture_promote(NvnTexture *t) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_PROMOTE") != NULL;
    if (off || !t || !t->addr || t->samples > 1 || t->target != 1 || g_npromo >= PROMO_MAX || tex_promoted(t)) return;
    g_promo[g_npromo].addr = t->addr; g_promo[g_npromo].w = t->width; g_promo[g_npromo].h = t->height;
    __atomic_store_n(&g_npromo, g_npromo + 1, __ATOMIC_RELEASE);
    LOG("[tex] promoted to the scene scale (rendered next to a scaled target of its size): addr %llx %ux%u f%#x flags %#x\n",
        (unsigned long long)t->addr, t->width, t->height, t->format, t->flags);
}
static void destroy_image(TexBackend *b);
static void create_image(NvnTexture *t, TexBackend *b);
// (exec.c begin_pass, outside a render pass) a promoted target whose object kept its unscaled backend (run436: the R8
// 1600x900 mask at 100968a000 is one long-lived object) is reallocated at the scene's scale, cleared to zero
bool renderer_texture_apply_promotion(NvnTexture *t) {
    TexBackend *b = t ? (TexBackend *)t->backend : NULL;
    if (!b || b->scalable || !g_npromo || !b->image || b->samples > 1 || b->blk || b->pending_rgba || !tex_promoted(t)) return false;
    extern uint32_t vk_view_gen;
    destroy_image(b);
    b->scale_kind = 1;
    b->scalable = true;
    create_image(t, b);
    vk_view_gen++;   // (exec.c's per-frame view memo is keyed by the texture object)
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkImageSubresourceRange rg = {b->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
    if (b->aspect & VK_IMAGE_ASPECT_COLOR_BIT) {
        VkClearColorValue z = {{0}};
        vkCmdClearColorImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg);
    } else {
        VkClearDepthStencilValue z = {0};
        vkCmdClearDepthStencilImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rg);
    }
    vk_full_barrier(cb);
    static int nlog;
    if (nlog++ < 32)
        LOG("[tex] promoted target reallocated: %p addr %llx %ux%u f%#x -> %ux%u\n", (void *)t, (unsigned long long)t->addr, t->width,
            t->height, t->format, b->width, b->height);
    return true;
}

// 0 = rendered at the game's size, 1 = 16:9 colour/depth target (output scale), 2 = shadow map
static int tex_scale_kind(const NvnTexture *t) {
    if (getenv("LARECOMP_NO_SCALE")) return 0;
    // (every 1920x1080 2D target, compressible or not, colour or depth: the inventory draws 6 targets at once, a compressible
    // one at the scene's scale next to the canvas's -> only the top-left 2/3 drawn, run424/425)
    if (out_scale_on() && t->samples <= 1 && t->target == 1 && t->width == 1920 && t->height == 1080) {
        static int nlog;
        if (nlog++ < 8) LOG("[tex] output canvas scale: addr %llx f%#x flags %#x\n", (unsigned long long)t->addr, t->format, t->flags);
        return 3;
    }
    if (t->samples <= 1 && t->target == 1 && g_npromo && tex_promoted(t)) return 1;
    if (t->samples > 1 || !(t->flags & 8)) return 0;
    if ((t->target == 1 || t->target == 4) && t->width == t->height && t->width >= 256 && renderer_format_is_depth(t->format) &&
        shadow_scale() != 1.0f)
        return 2;
    // 2D and 2D-array targets: a pass mixes them (TotK's 400x225 shadow-mask pass has a 2-layer array target next to 2D
    // ones), and begin_pass renders the minimum attachment extent, so an unscaled array clipped the pass to the top-left
    // 1/scale^2 of the screen (TOTK_RES_SCALE=2 lit only the top-left quarter: run333 tile flags 13x8 of 25x15)
    if ((t->target != 1 && t->target != 4) || t->width < 32 || t->height < 16) return 0;
    float a = (float)t->width / (float)t->height;
    return a > 1.74f && a < 1.81f ? 1 : 0;
}
static bool tex_scalable(const NvnTexture *t) { return tex_scale_kind(t) != 0; }
static float kind_sx(int kind) { return kind == 1 ? g_rs_x : kind == 2 ? shadow_scale() : kind == 3 ? g_out_s : 1.0f; }
static float kind_sy(int kind) { return kind == 1 ? g_rs_y : kind == 2 ? shadow_scale() : kind == 3 ? g_out_s : 1.0f; }

bool renderer_texture_first_write(NvnTexture *t) {
    TexBackend *b = t ? (TexBackend *)t->backend : NULL;
    uint64_t total_frames = renderer_present_count();
    if (!b || !b->presented || (b->sx == b->sy) || b->touched_frame == total_frames + 1) return false;
    b->touched_frame = total_frames + 1;
    return true;
}

static uint32_t g_level_gen = 1;
void vk_level_change_check(void) {
    extern const char *game_level_name(void);
    static char last[64];
    const char *lv = game_level_name();
    if (strncmp(lv, last, sizeof last - 1) != 0) {
        strlcpy(last, lv, sizeof last);
        if (!strncmp(lv, "TelephoneBox", 12) || getenv("LARECOMP_CLEAR_ALL_LEVELS")) g_level_gen++;
    }
}

bool renderer_texture_frame_fresh(NvnTexture *t) {
    static int on = -1;
    // TotK: OFF by default (LARECOMP_FRAME_CLEAR=1 = the LASRecomp behaviour). Real hardware never clears a target
    // implicitly; TotK redraws only some 32-px tiles of its half-res lighting targets per pass and keeps the rest, so
    // this clear left black tiles = the "squarey shadows" / patchy torch light (run128 pass_039 24.7 % covered,
    // run130 without the clear 100 %).
    if (on < 0) on = getenv("LARECOMP_FRAME_CLEAR") != NULL;
    TexBackend *b = t ? (TexBackend *)t->backend : NULL;
    uint64_t total_frames = renderer_present_count();
    if (!on || !b || !b->scalable || b->frame_cleared == total_frames + 1) return false;
    b->frame_cleared = total_frames + 1;
    return true;
}

bool renderer_texture_level_fresh(NvnTexture *t) {
    TexBackend *b = t ? (TexBackend *)t->backend : NULL;
    static int keep = -1;
    if (keep < 0) keep = getenv("LARECOMP_KEEP_TARGETS") != NULL;
    if (!b || keep || b->level_gen == g_level_gen) return false;
    if (!b->scalable) { b->level_gen = g_level_gen; return false; }
    bool fresh = b->level_gen != 0;
    b->level_gen = g_level_gen;
    return fresh;
}

void renderer_texture_mark_presented(NvnTexture *t) {
    TexBackend *b = t ? (TexBackend *)t->backend : NULL;
    if (b) b->presented = true;
}

bool renderer_texture_presented(NvnTexture *t) {
    TexBackend *b = t ? (TexBackend *)t->backend : NULL;
    return b && b->presented;
}

void renderer_texture_scale(NvnTexture *t, float *sx, float *sy) {
    TexBackend *b = t ? (TexBackend *)t->backend : NULL;
    if (b && b->scalable) { *sx = b->sx; *sy = b->sy; }
    else { *sx = 1; *sy = 1; }
}
static NvnTexture *alias_writer(NvnTexture *t);
// a sampled texture's scale = that of the image its view shows (vk_tex: another object's rendering at the same storage
// wins), not of the sampled object's own backend, which can be missing or of another kind
void renderer_texture_sample_scale(NvnTexture *t, float *sx, float *sy) {
    NvnTexture *w = t ? alias_writer(t) : NULL;
    renderer_texture_scale(w ? w : t, sx, sy);
}

// ================================================================== images
void vk_image_init_layout(VkCommandBuffer cb, VkImage img, VkImageAspectFlags aspect) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_NONE, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = img,
                               .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS},
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
}

// Images destroyed by the guest are kept until the submissions that may use them have completed.
typedef struct { VkImage img; VmaAllocation mem; VkImageView view; uint64_t seq; } Dead;
static Dead *dead;
static int ndead, cap_dead;
static void views_forget(TexBackend *b);

void vk_collect_dead(void) {
    uint64_t done = vk_completed_seq();
    int k = 0;
    for (int i = 0; i < ndead; i++) {
        if (dead[i].seq > done) { dead[k++] = dead[i]; continue; }
        if (dead[i].view) vkDestroyImageView(g_vk.dev, dead[i].view, NULL);
        else vmaDestroyImage(g_vk.vma, dead[i].img, dead[i].mem);
    }
    ndead = k;
}

uint64_t prof_img_n, prof_img_ns;  // (exec.c LARECOMP_VK_PROF) images created and the time spent creating them
uint64_t cum_img_n, cum_img_ns;    // the same, never reset (exec.c hitch log)
static void create_image_(NvnTexture *t, TexBackend *b);
static void create_image(NvnTexture *t, TexBackend *b) {
    extern uint64_t prof_up_ns;   // (~0: profiling off)
    uint64_t t0 = SDL_GetTicksNS();
    create_image_(t, b);
    uint64_t dt = SDL_GetTicksNS() - t0;
    __atomic_add_fetch(&cum_img_n, 1, __ATOMIC_RELAXED); __atomic_add_fetch(&cum_img_ns, dt, __ATOMIC_RELAXED);
    if (prof_up_ns != ~0ull) { prof_img_n++; prof_img_ns += dt; }
}
static void create_image_(NvnTexture *t, TexBackend *b) {
    VkFmt mf = fmt_of(t->format);
    b->sx = b->scalable ? kind_sx(b->scale_kind) : 1;
    b->sy = b->scalable ? kind_sy(b->scale_kind) : 1;
    uint32_t tw = t->width ? t->width : 1, th = t->height ? t->height : 1;
    if (b->scalable) {
        tw = (uint32_t)lroundf(tw * b->sx); th = (uint32_t)lroundf(th * b->sy);
        if (tw < 1) tw = 1;
        if (th < 1) th = 1;
    }
    VkImageType type = VK_IMAGE_TYPE_2D;
    uint32_t layers = 1, depth = 1;
    switch (t->target) {
    case 0: type = VK_IMAGE_TYPE_1D; th = 1; break;
    case 2: type = VK_IMAGE_TYPE_3D; depth = t->depth ? t->depth : 1; break;
    case 3: type = VK_IMAGE_TYPE_1D; layers = t->height ? t->height : 1; th = 1; break;
    case 4: case 6: layers = t->depth ? t->depth : 1; break;
    case 8: layers = 6; break;
    case 9: layers = t->depth ? t->depth : 6; break;
    default: break;
    }
    uint32_t levels = t->levels ? t->levels : 1;
    {   // never more levels than the size allows (scaled targets, odd guest values)
        uint32_t m = 1, sz = tw > th ? tw : th;
        if (depth > sz) sz = depth;
        while (sz > 1) { sz >>= 1; m++; }
        if (levels > m) levels = m;
    }
    uint32_t samples = t->samples > 1 ? t->samples : 1;
    if (samples > 1) levels = 1;
    VkFormat vf = mf.vf;
    VkFormatFeatureFlags need = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    b->astc_bw = b->astc_bh = 0;
    if (!format_supports(vf, need)) {
        bool astc = vf >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK && vf <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
        const FormatInfo *fi = t->layout.fmt;
        if (astc && fi && fi->bpb == 16) {  // desktop GPUs: decoded on the CPU into an RGBA8 image of the same colour space
            b->astc_bw = fi->bw;
            b->astc_bh = fi->bh;
            b->astc_srgb = t->format >= 0x87;
            vf = b->astc_srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        } else {
            static int nlog;
            if (nlog++ < 20) LOG("[vk] format %d (nvn %#x) not supported: RGBA8 instead\n", vf, t->format);
            vf = VK_FORMAT_R8G8B8A8_UNORM;
        }
    }
    bool depthfmt = renderer_format_is_depth(t->format);
    VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (depthfmt) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    else if (format_supports(vf, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    // (render targets the game also samples while they are bound: exec.c declares those feedback loops, device.c)
    if (g_vk.feedback_loop && (usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)))
        usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    VkImageCreateFlags flags = 0;
    if (!depthfmt) flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;  // format views (NVN texture views, raw sRGB views)
    if (type == VK_IMAGE_TYPE_2D && layers >= 6 && tw == th && samples == 1) flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    // 3D render targets (TotK renders mips / slices of 3D volumes, run285): slices are drawn through 2D views
    if (type == VK_IMAGE_TYPE_3D && (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) flags |= VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = flags, .imageType = type, .format = vf,
                            .extent = {tw, th, depth}, .mipLevels = levels, .arrayLayers = layers,
                            .samples = (VkSampleCountFlagBits)samples, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    extern _Thread_local int vma_shim_tag;
    vma_shim_tag = b->astc_bw ? 1 : 2;
    VkResult r = vmaCreateImage(g_vk.vma, &ci, &ai, &b->image, &b->mem, NULL);
    if (r != VK_SUCCESS) {
        LOG("[vk] image creation failed (%d): nvn fmt %#x %ux%ux%u target %u\n", r, t->format, t->width, t->height, t->depth, t->target);
        ci.format = vf = VK_FORMAT_R8G8B8A8_UNORM;
        ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        ci.flags &= ~VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
        VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &b->image, &b->mem, NULL));
        depthfmt = false;
    }
    vma_shim_tag = 0;
    b->format = vf;
    b->usage = ci.usage;
    b->cflags = ci.flags;
    b->width = tw; b->height = th; b->depth = depth; b->layers = layers; b->levels = levels; b->samples = samples;
    b->aspect = depthfmt ? aspect_of(t->format) : VK_IMAGE_ASPECT_COLOR_BIT;
    b->base_type = type == VK_IMAGE_TYPE_1D ? (layers > 1 ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D)
                 : type == VK_IMAGE_TYPE_3D ? VK_IMAGE_VIEW_TYPE_3D
                 : t->target == 8 ? VK_IMAGE_VIEW_TYPE_CUBE : t->target == 9 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY
                 : layers > 1 || t->target == 4 || t->target == 6 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    vk_image_init_layout(vk_upload_cmd(), b->image, b->aspect);
    if (depthfmt) {
        // New depth / stencil images start zeroed (as fresh memory reads on Apple GPUs). The game builds its depth
        // pyramid with depth-only copies, so the stencil of those images was never written: whatever the memory
        // held, different per image and per run, then failed or passed the depth-of-field passes' stencil test in
        // patches (flicker that came and went between runs, every other frame, below 1x internal resolution).
        VkCommandBuffer cb = vk_upload_cmd();
        VkClearDepthStencilValue zero = {0.0f, 0};
        VkImageSubresourceRange rg = {b->aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
        vkCmdClearDepthStencilImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rg);
        VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
        vkCmdPipelineBarrier2(cb, &di);
    }
}

static void defer(Dead d) {
    if (ndead == cap_dead) { cap_dead = cap_dead ? cap_dead * 2 : 256; dead = realloc(dead, cap_dead * sizeof *dead); }
    d.seq = vk_pending_seq();
    dead[ndead++] = d;
}

void vk_defer_image(VkImage img, VmaAllocation mem) { defer((Dead){.img = img, .mem = mem}); }
void vk_defer_view(VkImageView v) { defer((Dead){.view = v}); }

void vk_dl_note_upload(void *img, uint32_t w, uint32_t h, uint32_t lv);  // (exec.c: LARECOMP_VK_DEVLOST_TRACE)

static uint32_t level_w(const TexBackend *b, uint32_t lv) { uint32_t v = b->width >> lv; return v ? v : 1; }
static uint32_t level_h(const TexBackend *b, uint32_t lv) { uint32_t v = b->height >> lv; return v ? v : 1; }

static void destroy_image(TexBackend *b) {
    views_forget(b);
    if (b->image || b->mem) defer((Dead){.img = b->image, .mem = b->mem});  // (never the same allocation twice)
    b->image = VK_NULL_HANDLE;
    b->mem = VK_NULL_HANDLE;
    if (b->blk) { destroy_image(b->blk); free(b->blk); b->blk = NULL; }
}

// ---- block-compressed render targets (run288: TotK fills 64^3 / 32^3 BC4 volumes by rendering ENCODED blocks through an
// RGBA16UI view, one texel per 4x4 block). Vulkan cannot render to a BC image, so the pass draws into an uncompressed
// "block shadow" image (w/4 x h/4 x depth, the view's format) and exec.c copies each rendered level/slice into the BC
// image after the pass (vkCmdCopyImage between size-compatible formats). LARECOMP_NO_BC_RT=1: dropped (not rendered).
static bool is_bc(VkFormat f) { return f >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && f <= VK_FORMAT_BC7_SRGB_BLOCK; }
static uint32_t bc_block_bytes(VkFormat f) {
    return f <= VK_FORMAT_BC1_RGBA_SRGB_BLOCK || f == VK_FORMAT_BC4_UNORM_BLOCK || f == VK_FORMAT_BC4_SNORM_BLOCK ? 8 : 16;
}
static uint32_t texel_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R16G16B16A16_UINT: case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_SFLOAT:
    case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    default: return 0;
    }
}

static TexBackend *block_shadow(NvnTexture *t, TexBackend *b, VkFormat vf) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_NO_BC_RT") != NULL;
    if (off || !texel_bytes(vf) || texel_bytes(vf) != bc_block_bytes(b->format) || b->samples > 1) return NULL;
    if (b->blk && b->blk->format == vf) return b->blk;
    if (b->blk) { destroy_image(b->blk); free(b->blk); b->blk = NULL; }
    TexBackend *s = calloc(1, sizeof *s);
    bool is3d = b->base_type == VK_IMAGE_VIEW_TYPE_3D;
    uint32_t w = (b->width + 3) / 4, h = (b->height + 3) / 4, d = is3d ? b->depth : 1;
    uint32_t m = 1, sz = w > h ? w : h;
    if (d > sz) sz = d;
    while (sz > 1) { sz >>= 1; m++; }
    uint32_t levels = b->levels < m ? b->levels : m;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                            .flags = is3d ? VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT : 0,
                            .imageType = is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D, .format = vf, .extent = {w, h, d},
                            .mipLevels = levels, .arrayLayers = is3d ? 1 : b->layers, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    extern _Thread_local int vma_shim_tag;
    vma_shim_tag = 3;
    VkResult sr_ = vmaCreateImage(g_vk.vma, &ci, &ai, &s->image, &s->mem, NULL);
    vma_shim_tag = 0;
    if (sr_ != VK_SUCCESS) {
        LOG("[vk] block shadow creation failed: %ux%ux%u vk fmt %d\n", w, h, d, vf);
        free(s);
        return NULL;
    }
    s->format = vf; s->usage = ci.usage; s->cflags = ci.flags;
    s->width = w; s->height = h; s->depth = d; s->layers = ci.arrayLayers; s->levels = levels; s->samples = 1;
    s->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    s->base_type = is3d ? VK_IMAGE_VIEW_TYPE_3D : b->base_type;
    s->gpu_owned = true;
    vk_image_init_layout(vk_upload_cmd(), s->image, s->aspect);
    b->blk = s;
    LOG("[vk] block shadow for BC render target %ux%ux%u vk fmt %d addr %llx: %ux%ux%u vk fmt %d, %u levels\n", t->width,
        t->height, t->depth, b->format, (unsigned long long)t->addr, w, h, d, vf, levels);
    return s;
}

TexBackend *vk_render_backend(NvnTexture *t) {
    TexBackend *b = t->backend;
    return b && b->blk && is_bc(b->format) ? b->blk : b;
}

// Live game textures: an in-place reboot (vk_textures_reset) destroys them all.
static NvnTexture **g_live;
static int g_nlive, g_cap_live;
static void live_add(NvnTexture *t, TexBackend *b) {
    if (g_nlive == g_cap_live) { g_cap_live = g_cap_live ? g_cap_live * 2 : 1024; g_live = realloc(g_live, g_cap_live * sizeof *g_live); }
    g_live[g_nlive++] = t;
    b->reg = g_nlive;
}
static void live_remove(TexBackend *b) {
    if (!b->reg) return;
    NvnTexture *last = g_live[--g_nlive];
    g_live[b->reg - 1] = last;
    if (last->backend) ((TexBackend *)last->backend)->reg = b->reg;
    b->reg = 0;
}

// ---- image recycling. TotK initializes and finalizes ~30 transient render textures per frame (run107: 1812 images
// created per 2 s, 315 ms in vmaCreateImage, plus their destruction and new views - most of begin_pass's 850 ms).
// A finalized texture's image goes to a pool; a new texture of the same shape takes it with its views. Contents are
// undefined, as with reused pool memory on hardware. LARECOMP_VK_NO_RECYCLE=1 disables.
#define RECYCLE_MAX 512
typedef struct { TexBackend *b; uint32_t format, target, w, h, d, levels, samples; bool scalable; float sx, sy; uint64_t seq, addr; bool gpu_owned; } Recycled;
static Recycled g_recycle[RECYCLE_MAX];
static int g_nrecycle;
static uint64_t g_recycle_bytes;   // (device memory held by pooled images: "[vram]" line)
uint64_t vk_recycle_bytes(int *n) { if (n) *n = g_nrecycle; return g_recycle_bytes; }
VkDeviceSize vma_shim_size(VmaAllocation al);
static pthread_mutex_t g_recycle_lock = PTHREAD_MUTEX_INITIALIZER;
uint64_t prof_img_reused, prof_img_same_addr;
static int recycle_off(void) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_RECYCLE") != NULL;
    return off;
}
static void free_backend(TexBackend *b) { destroy_image(b); free(b->pending_rgba); free(b); }
static bool recycle_put(NvnTexture *t, TexBackend *b) {
    if (recycle_off() || !b->image || b->pending_rgba) return false;
    extern uint32_t vk_view_gen;
    vk_view_gen++;   // exec.c's per-frame view memo is keyed by the NvnTexture pointer, which the guest may reuse
    // Byte cap (run574: the pool held up to 993 MB of 3200x1800 targets, many of sizes the game's dynamic resolution had
    // left; with ~4 GB of mirrors that pushed VRAM over the OS budget -> 3-14 fps in run571). Oldest go first.
    // LARECOMP_VK_RECYCLE_MB=<n> (default 512; 0 = no cap).
    static int64_t cap = -1;
    if (cap < 0) { const char *e = getenv("LARECOMP_VK_RECYCLE_MB"); cap = (int64_t)(e ? atol(e) : 512) << 20; }
    TexBackend *evict[8];
    int nev = 0;
    pthread_mutex_lock(&g_recycle_lock);
    g_recycle_bytes += vma_shim_size(b->mem);
    while (g_nrecycle && nev < 8 && (g_nrecycle == RECYCLE_MAX || (cap && g_recycle_bytes > (uint64_t)cap))) {
        evict[nev] = g_recycle[0].b;
        memmove(g_recycle, g_recycle + 1, (size_t)(g_nrecycle - 1) * sizeof *g_recycle);
        g_nrecycle--;
        g_recycle_bytes -= vma_shim_size(evict[nev++]->mem);
    }
    g_recycle[g_nrecycle++] = (Recycled){b, t->format, t->target, t->width, t->height, t->depth, t->levels, t->samples, b->scalable, b->sx, b->sy,
                                         vk_pending_seq(), t->addr, b->gpu_owned};
    pthread_mutex_unlock(&g_recycle_lock);
    for (int i = 0; i < nev; i++) free_backend(evict[i]);
    return true;
}
static bool g_recycle_same;   // (debug log: the last recycle_get kept the same storage's image)
static bool recycle_shape(const Recycled *r, const NvnTexture *t, bool sc) {
    return r->format == t->format && r->target == t->target && r->w == t->width && r->h == t->height && r->d == t->depth &&
           r->levels == t->levels && r->samples == t->samples && r->scalable == sc &&
           (!sc || (r->sx == kind_sx(tex_scale_kind(t)) && r->sy == kind_sy(tex_scale_kind(t))));
}
static TexBackend *recycle_get(NvnTexture *t) {
    if (recycle_off()) return NULL;
    // Same storage first: an NVN texture is a view of pool memory, so a new object over the storage of a finalized one
    // sees what that one rendered. TotK re-creates its render targets every frame (run245: images were handed to other
    // addresses of the same shape) and reads the previous frame's results through the new objects. The image used at
    // this address with this shape comes back with its contents and GPU ownership (no upload of the stale guest copy).
    // Other shapes/addresses take the OLDEST pooled image, leaving recent ones to their own storage.
    // LARECOMP_VK_RECYCLE_ANY=1 = the old shape-only, newest-first reuse.
    static int any = -1;
    if (any < 0) any = getenv("LARECOMP_VK_RECYCLE_ANY") != NULL;
    bool sc = tex_scalable(t);
    TexBackend *b = NULL;
    bool same = false, owned = false;
    uint64_t cur = vk_pending_seq();
    pthread_mutex_lock(&g_recycle_lock);
    int pick = -1;
    if (!any && t->addr)
        for (int i = g_nrecycle - 1; i >= 0; i--) {
            Recycled *r = &g_recycle[i];
            if (r->addr == t->addr && recycle_shape(r, t, sc) && (r->gpu_owned || r->seq < cur)) { pick = i; same = true; owned = r->gpu_owned; break; }
        }
    if (pick < 0) {
        for (int k = 0; k < g_nrecycle; k++) {
            int i = any ? g_nrecycle - 1 - k : k;   // (old behaviour: newest first)
            Recycled *r = &g_recycle[i];
            if (!recycle_shape(r, t, sc) ||
                r->seq >= cur)   // finalized in the open submission: its last uses may still be recorded after our uploads run
                continue;
            pick = i;
            break;
        }
    }
    if (pick >= 0) {
        b = g_recycle[pick].b;
        g_recycle_bytes -= vma_shim_size(b->mem);
        memmove(&g_recycle[pick], &g_recycle[pick + 1], (size_t)(g_nrecycle - 1 - pick) * sizeof *g_recycle);
        g_nrecycle--;
    }
    pthread_mutex_unlock(&g_recycle_lock);
    if (!b) return NULL;
    prof_img_reused++;
    if (same) {   // same memory: keep contents and state (only the live-list slot is new)
        b->reg = 0;   // (write-back state kept: a target re-created over the same storage every frame stays "per-frame";
                      // run377: a 144x144 R8 target at 100762b000 was written back every frame, 2.9 ms GPU wait each)
        b->zero_on_write = false;
        b->gpu_owned = owned;
        prof_img_same_addr++;
        g_recycle_same = true;
        return b;
    }
    // fresh texture state on the same image and views
    b->wb_frame = 0; b->wb_state = 0;
    b->gpu_owned = b->uploaded = b->dynamic = b->cpu_dirty = b->presented = false;
    b->data_hash = b->sample_hash = b->checked_frame = b->full_frame = b->touched_frame = b->frame_cleared = 0;
    b->level_gen = 0; b->tp = 0; b->tp_gen = 0; b->reg = 0;
    // order against the previous owner's GPU work (earlier submissions) before this submission's uploads write it
    vk_full_barrier(vk_upload_cmd());
    // depth / stencil start zeroed as new images do (create_image_): cleared by the first pass that renders to it
    b->zero_on_write = (b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
    return b;
}
static void recycle_flush(void) {
    pthread_mutex_lock(&g_recycle_lock);
    int n = g_nrecycle;
    g_nrecycle = 0;
    g_recycle_bytes = 0;
    pthread_mutex_unlock(&g_recycle_lock);
    for (int i = 0; i < n; i++) free_backend(g_recycle[i].b);
}

bool tex_watched(uint64_t addr);
static TexBackend *tex_backend(NvnTexture *t) {
    TexBackend *b = t->backend;
    if (b) {
        if (b->scalable && (b->sx != kind_sx(b->scale_kind) || b->sy != kind_sy(b->scale_kind))) {  // output resolution changed: reallocate
            destroy_image(b);
            create_image(t, b);
        }
        return b;
    }
    g_recycle_same = false;
    b = recycle_get(t);
    bool reused = b != NULL;
    if (!b) {
        vk_collect_dead();
        b = calloc(1, sizeof *b);
        b->scale_kind = (uint8_t)tex_scale_kind(t);
        b->scalable = b->scale_kind != 0;
        create_image(t, b);
    }
    t->backend = b;
    live_add(t, b);
    extern bool g_debug_frame;
    if (g_debug_frame || tex_watched(t->addr))
        LOG("[tex] backend for %p addr %llx %ux%u f%#x: %s image %p\n", (void *)t, (unsigned long long)t->addr, t->width, t->height,
            t->format, reused ? (g_recycle_same ? "same-storage" : "recycled") : "new", (void *)b->image);
    return b;
}

// ---- aliases: NVN textures are views of pool memory. TotK renders an image through one texture object and samples
// it through another over the same storage (run101: the FXAA input, never a render target itself, read all zero:
// it was uploaded from guest memory, which GPU rendering never writes). The last texture object rendered to at an
// address is its writer; a sampled object with the same address, size and texel size uses the writer's image.
// LARECOMP_VK_NO_ALIAS=1 disables.
#define ALIAS_TAB 4096
static struct { uint64_t addr; NvnTexture *t; } g_alias[ALIAS_TAB];
static uint32_t alias_slot(uint64_t addr) { return (uint32_t)((addr >> 8) * 0x9E3779B97F4A7C15ull >> 52) & (ALIAS_TAB - 1); }
static void alias_note_writer(NvnTexture *t) {
    if (!t->addr) return;
    uint32_t s = alias_slot(t->addr);
    int freeslot = -1;
    for (int p = 0; p < 32; p++, s = (s + 1) & (ALIAS_TAB - 1)) {
        if (g_alias[s].addr == t->addr) { g_alias[s].t = t; return; }
        if (!g_alias[s].t && freeslot < 0) freeslot = (int)s;   // empty or forgotten
        if (!g_alias[s].addr) break;
    }
    if (freeslot >= 0) { g_alias[freeslot].addr = t->addr; g_alias[freeslot].t = t; }
}
static void alias_forget(NvnTexture *t) {
    if (!t->addr) return;
    uint32_t s = alias_slot(t->addr);
    for (int p = 0; p < 32; p++, s = (s + 1) & (ALIAS_TAB - 1))
        if (g_alias[s].t == t) { g_alias[s].t = NULL; return; }   // (slot kept: probe chains stay intact; reused by the next writer)
}
static NvnTexture *alias_writer(NvnTexture *t) {
    static int off = -1;
    if (off < 0) off = getenv("LARECOMP_VK_NO_ALIAS") != NULL;
    if (off || !t->addr) return NULL;
    uint32_t s = alias_slot(t->addr);
    for (int p = 0; p < 32; p++, s = (s + 1) & (ALIAS_TAB - 1)) {
        if (g_alias[s].addr != t->addr) { if (!g_alias[s].addr) return NULL; continue; }
        NvnTexture *w = g_alias[s].t;
        if (!w || w == t || !w->backend || !((TexBackend *)w->backend)->gpu_owned) return NULL;
        const FormatInfo *fa = t->layout.fmt, *fb = w->layout.fmt;
        if (w->width != t->width || w->height != t->height || w->target != t->target || w->samples != t->samples || !fa || !fb ||
            fa->bpb != fb->bpb || fa->bw != fb->bw || fa->bh != fb->bh ||
            renderer_format_is_depth(w->format) != renderer_format_is_depth(t->format))
            return NULL;
        // one line per distinct sampled object (open-addressed set, <= 8 probes: this runs per sampled texture per draw;
        // the old linear scan of up to 256 entries was 2.2 % of the render thread, run590)
        static NvnTexture *logged[512];
        uint32_t hs = (uint32_t)((uintptr_t)t >> 4) * 0x9E3779B1u >> 23;
        for (int i = 0; i < 8; i++) {
            NvnTexture **e = &logged[(hs + i) & 511];
            if (*e == t) break;
            if (!*e) {
                *e = t;
                LOG("[tex] alias: %p (%ux%u f%#x) samples writer %p (f%#x) at %llx\n", (void *)t, t->width, t->height, t->format,
                    (void *)w, w->format, (unsigned long long)t->addr);
                break;
            }
        }
        return w;
    }
    return NULL;
}

bool renderer_texture_gpu_owned(NvnTexture *t) {
    const TexBackend *b = t ? (const TexBackend *)t->backend : NULL;
    return b && b->gpu_owned;
}
// another object's live rendering at t's storage (same size and texel layout) would stand in when t is sampled
bool renderer_texture_has_writer(NvnTexture *t) { return alias_writer(t) != NULL; }
NvnTexture *renderer_texture_alias_writer(NvnTexture *t) { return alias_writer(t); }   // (exec.c resolve_feedback)

static void wb_forget(NvnTexture *t);
void backend_texture_destroyed(NvnTexture *t) {
    alias_forget(t);
    wb_forget(t);
    TexBackend *b = t->backend;
    extern bool g_debug_frame;
    if ((g_debug_frame || tex_watched(t->addr)) && b)
        LOG("[tex] finalize %p addr %llx %ux%u f%#x image %p\n", (void *)t, (unsigned long long)t->addr, t->width, t->height, t->format,
            (void *)b->image);
    if (b) {
        live_remove(b);
        if (!recycle_put(t, b)) free_backend(b);
    }
    t->backend = NULL;
}

// Reboot (device.c backend_reset; GPU idle): every game texture and its views go.
void vk_textures_reset(void) {
    int n = g_nlive;
    while (g_nlive > 0) backend_texture_destroyed(g_live[g_nlive - 1]);
    recycle_flush();
    vk_collect_dead();
    LOG("[vk] reboot: %d textures released\n", n);
}

// ================================================================== uploads (same change detection as metal.m)
static uint64_t full_hash_budget, full_hash_budget_frame;

// Four independent lanes (one serial multiply chain ran at ~1.5 bytes/cycle; dynamic textures are hashed every
// frame). Values are only compared within a run.
static uint64_t hash_full(const uint8_t *p, uint64_t n) {
    uint64_t h0 = 0x9E3779B97F4A7C15ull ^ n, h1 = 0xC2B2AE3D27D4EB4Full, h2 = 0x165667B19E3779F9ull, h3 = 0x27D4EB2F165667C5ull;
    uint64_t i = 0;
    for (; i + 32 <= n; i += 32) {
        uint64_t v[4];
        memcpy(v, p + i, 32);
        h0 = (h0 ^ v[0]) * 0x100000001B3ull; h0 ^= h0 >> 29;
        h1 = (h1 ^ v[1]) * 0x100000001B3ull; h1 ^= h1 >> 29;
        h2 = (h2 ^ v[2]) * 0x100000001B3ull; h2 ^= h2 >> 29;
        h3 = (h3 ^ v[3]) * 0x100000001B3ull; h3 ^= h3 >> 29;
    }
    uint64_t h = h0 ^ (h1 * 0x9E3779B97F4A7C15ull) ^ (h2 * 0xC2B2AE3D27D4EB4Full) ^ (h3 * 0x165667B19E3779F9ull);
    for (; i < n; i++) h = (h ^ p[i]) * 0x100000001B3ull;
    return h ^ (h >> 31);
}

static uint64_t fnv_sample(const uint8_t *p, uint64_t n) {
    uint64_t h = 1469598103934665603ull;
    uint64_t step = n > 4096 ? n / 4096 : 1;
    for (uint64_t i = 0; i < n; i += step) { h ^= p[i]; h *= 1099511628211ull; }
    return h ^ n;
}

static pthread_mutex_t pending_lock = PTHREAD_MUTEX_INITIALIZER;

static void upload_pending(TexBackend *b) {
    pthread_mutex_lock(&pending_lock);
    void *px = b->pending_rgba;
    uint32_t w = b->pending_w, h = b->pending_h, stride = b->pending_stride;
    b->pending_rgba = NULL;
    pthread_mutex_unlock(&pending_lock);
    if (!px) return;
    if (w > b->width) w = b->width;
    if (h > b->height) h = b->height;
    vk_up_kind = 4; VkSpan s = vk_transient_copy(px, (uint64_t)stride * h, 16); vk_up_kind = 0;
    free(px);
    VkBufferImageCopy rg = {.bufferOffset = s.off, .bufferRowLength = stride / 4,
                            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
    VkCommandBuffer cb = vk_upload_cmd();
    vk_dl_note_upload((void *)b->image, rg.imageExtent.width, rg.imageExtent.height, rg.imageSubresource.mipLevel);
                vkCmdCopyBufferToImage(cb, s.buf, b->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
}

uint64_t prof_tex_checks, prof_tex_samples, prof_tex_full, prof_tex_full_bytes, prof_tex_dyn, prof_tex_ww_clean;
// (LARECOMP_VK_PROF) full-hashed textures: "[texhash]" every 600 frames lists the 10 with the most hashed bytes, with how
// often their contents had changed and the hash / upload time (are "dynamic" textures really rewritten every frame?)
extern uint64_t prof_up_ns;
static struct { TexBackend *b; uint64_t addr, bytes, hashes, changes, hash_ns, up_ns; uint32_t w, h, fmt, flags; } th[64];
static uint64_t th_frame;
static void texhash_note(NvnTexture *t, TexBackend *b, uint64_t total, bool changed, uint64_t hash_ns, uint64_t up_ns) {
    int i = 0;
    while (i < 64 && th[i].b && th[i].b != b) i++;
    if (i == 64) return;
    if (!th[i].b) { th[i].b = b; th[i].addr = t->addr; th[i].w = t->width; th[i].h = t->height; th[i].fmt = t->format; th[i].flags = t->flags; }
    th[i].bytes += total; th[i].hashes++; th[i].changes += changed; th[i].hash_ns += hash_ns; th[i].up_ns += up_ns;
}
static void texhash_report(uint64_t frame) {
    if (!th_frame) th_frame = frame;
    if (frame - th_frame < 600) return;
    LOG("[texhash] frames %llu..%llu, top full-hashed textures:\n", (unsigned long long)th_frame, (unsigned long long)frame);
    for (int k = 0; k < 10; k++) {
        int best = -1;
        for (int i = 0; i < 64; i++) if (th[i].b && th[i].hashes && (best < 0 || th[i].bytes > th[best].bytes)) best = i;
        if (best < 0) break;
        LOG("[texhash]   %llx %ux%u fmt %#x flags %#x: %llu hashes, %llu changed, %.0f MB, hash %.1f ms, upload %.1f ms\n",
            (unsigned long long)th[best].addr, th[best].w, th[best].h, th[best].fmt, th[best].flags, (unsigned long long)th[best].hashes,
            (unsigned long long)th[best].changes, th[best].bytes / 1048576.0, th[best].hash_ns / 1e6, th[best].up_ns / 1e6);
        th[best].hashes = 0;
    }
    memset(th, 0, sizeof th);
    th_frame = frame;
}

bool tex_watched(uint64_t addr);
static void tex_watch_note(NvnTexture *t, TexBackend *b, const char *what);

static void upload_texture(NvnTexture *t, TexBackend *b) {
    if (!t->addr || t->target == 10) return;
    uint64_t frame = vk_frame_counter();
    bool cpu_dirty = b->cpu_dirty;
    b->cpu_dirty = false;
    if (b->uploaded && b->checked_frame == frame && !cpu_dirty) return;
    if (b->uploaded && !b->dynamic && ((frame + ((uintptr_t)b >> 6)) & 7) && !cpu_dirty) return;
    const FormatInfo *fi = t->layout.fmt;
    if (!fi || !fi->bpb) return;
    VkFmt mf = fmt_of(t->format);
    uint64_t total = tegra_total_size(&t->layout);
    b->checked_frame = frame;
    prof_tex_checks++;
    if (t->flags & 0x10) b->dynamic = true;
    if (full_hash_budget_frame != frame) { full_hash_budget_frame = frame; full_hash_budget = 4 << 20; }
    const uint8_t *mem = (const uint8_t *)(uintptr_t)t->addr;
    static int always_full = -1;
    if (always_full < 0) always_full = getenv("LARECOMP_TEX_FULLHASH") ? 1 : 0;
    bool full = b->dynamic || !b->uploaded || always_full || cpu_dirty;
    // Guest heap write watch (hle_os_mem.cpp heap_ww_changed): an uploaded texture whose pages the CPU has not written
    // since its last check is unchanged, without hashing. GPU writes into the heap are invisible to the watch, so a full
    // hash still runs every 60 frames (dynamic) / 120 frames (others). TOTK_HEAP_WW=0: off.
    if (b->uploaded && !cpu_dirty && !always_full && frame - b->full_frame < (b->dynamic ? 60u : 120u)) {
        extern int heap_ww_changed(uint64_t addr, uint64_t size, uint64_t *seen);
        int ch = heap_ww_changed(t->addr, total, &b->ww_seen);
        if (ch == 0 && !(b->tp > 0 && b->tp_gen != texpatch_generation())) { prof_tex_ww_clean++; return; }
        if (ch > 0) full = true;   // written: hash it now (dynamic or not)
    }
    if (!full && frame - b->full_frame >= 120 && full_hash_budget >= total) { full = true; full_hash_budget -= total; }
    if (!b->tp) b->tp = texpatch_identify(t) + 1;  // (button prompts / Hero Mode icon: re-uploaded when their look changes)
    bool restyle = b->tp > 0 && b->uploaded && b->tp_gen != texpatch_generation();
    if (!full && !restyle) {
        prof_tex_samples++;
        uint64_t sh = fnv_sample(mem, total);
        if (sh == b->sample_hash) return;
    }
    prof_tex_full++; prof_tex_full_bytes += total; if (b->dynamic) prof_tex_dyn++;
    const bool tprof = prof_up_ns != ~0ull;
    uint64_t th0 = tprof ? SDL_GetTicksNS() : 0;
    uint64_t h = hash_full(mem, total);
    b->full_frame = frame;
    b->sample_hash = fnv_sample(mem, total);
    uint64_t th1 = tprof ? SDL_GetTicksNS() : 0;
    if (tprof) texhash_report(frame);
    if (b->uploaded && h == b->data_hash && !restyle) { if (tprof) texhash_note(t, b, total, false, th1 - th0, 0); return; }
    if (tprof) texhash_note(t, b, total, b->uploaded, th1 - th0, 0);
    if (b->uploaded && h != b->data_hash) b->dynamic = true;
    if (b->tp > 0) b->tp_gen = texpatch_generation();
    b->data_hash = h;
    b->uploaded = true;
    if (tex_watched(t->addr)) {
        uint64_t nz = 0;
        for (uint64_t i = 0; i < total; i++) nz += mem[i] != 0;
        char what[64];
        snprintf(what, sizeof what, "CPU upload %llu/%llu bytes nonzero", (unsigned long long)nz, (unsigned long long)total);
        tex_watch_note(t, b, what);
        LOG("[texw]   flags %#x stride %u target %u levels %u layout bh_log2 %u bd_log2 %u layers %u\n", t->flags, t->stride, t->target,
            t->layout.levels, t->layout.block_height_log2, t->layout.block_depth_log2, t->layout.layers);
        char path[96];   // raw guest bytes (offline: try linear / other block shapes, tools side)
        snprintf(path, sizeof path, "build/texw_%llx.bin", (unsigned long long)t->addr);
        FILE *f = fopen(path, "wb");
        if (f) { fwrite(mem, 1, total, f); fclose(f); LOG("[texw]   raw bytes -> %s\n", path); }
    }
    if (b->format != mf.vf && !b->astc_bw) return;  // unsupported format: nothing sensible to copy
    VkCommandBuffer cb = vk_upload_cmd();
    uint32_t layers = t->layout.layers ? t->layout.layers : 1;
    if (layers > b->layers && b->depth == 1) layers = b->layers;
    uint64_t layer_size = tegra_layer_size(&t->layout);
    bool linear = nvn_flags_linear(t->flags);
    uint32_t levels = t->layout.levels < b->levels ? t->layout.levels : b->levels;
    for (uint32_t layer = 0; layer < layers; layer++) {
        for (uint32_t lv = 0; lv < levels; lv++) {
            LevelLayout l = tegra_level(&t->layout, lv);
            if (b->astc_bw) {  // blocks -> linear block rows -> RGBA8 texels
                const uint8_t *src = (const uint8_t *)(uintptr_t)t->addr + layer * layer_size + l.offset;
                uint8_t *blocks = malloc((uint64_t)l.wb * l.hb * 16);
                if (linear) for (uint32_t y = 0; y < l.hb; y++) memcpy(blocks + (uint64_t)y * l.wb * 16, src + (uint64_t)y * t->stride, l.wb * 16);
                else tegra_deswizzle(src, blocks, l.wb, l.hb, 1, 16, l.block_height_log2, l.block_depth_log2);
                if (b->tp > 0 && layer == 0) texpatch_apply(b->tp - 1, lv, blocks, l.w, l.h, l.wb * 16);
                vk_up_kind = 4; VkSpan s = vk_transient((uint64_t)l.w * l.h * 4, 16); vk_up_kind = 0;
                astc_decode_ldr(blocks, l.w, l.h, b->astc_bw, b->astc_bh, b->astc_srgb, s.ptr);
                free(blocks);
                VkBufferImageCopy rg = {.bufferOffset = s.off, .bufferRowLength = l.w, .bufferImageHeight = l.h,
                                        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, layer, 1},
                                        .imageExtent = {l.w < level_w(b, lv) ? l.w : level_w(b, lv), l.h < level_h(b, lv) ? l.h : level_h(b, lv), 1}};
                vk_dl_note_upload((void *)b->image, rg.imageExtent.width, rg.imageExtent.height, rg.imageSubresource.mipLevel);
                vkCmdCopyBufferToImage(cb, s.buf, b->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
                continue;
            }
            uint32_t row = l.wb * fi->bpb;
            uint64_t sz = (uint64_t)row * l.hb * l.d;
            bool expand = mf.expand_rgb && fi->bpb == 3;
            uint32_t out_row = expand ? l.wb * 4 : row;
            vk_up_kind = 4; VkSpan s = vk_transient((uint64_t)out_row * l.hb * l.d, 16); vk_up_kind = 0;
            uint8_t *lin = expand ? malloc(sz) : s.ptr;
            const uint8_t *src = (const uint8_t *)(uintptr_t)t->addr + layer * layer_size + l.offset;
            if (linear) {
                for (uint32_t y = 0; y < l.hb; y++) memcpy(lin + (uint64_t)y * row, src + (uint64_t)y * t->stride, row);
            } else {
                tegra_deswizzle(src, lin, l.wb, l.hb, l.d, fi->bpb, l.block_height_log2, l.block_depth_log2);
            }
            if (b->tp > 0 && layer == 0) texpatch_apply(b->tp - 1, lv, lin, l.w, l.h, row);
            if (expand) {  // RGB8 -> RGBA8
                uint8_t *o = s.ptr;
                uint64_t n = (uint64_t)l.wb * l.hb * l.d;
                for (uint64_t i = 0; i < n; i++) { o[i * 4] = lin[i * 3]; o[i * 4 + 1] = lin[i * 3 + 1]; o[i * 4 + 2] = lin[i * 3 + 2]; o[i * 4 + 3] = 255; }
                free(lin);
            }
            uint32_t w = l.w, hh = l.h, d = l.d;
            // the copy stays inside the image's level: a texture allocated at the internal resolution (16:9, render
            // target sized) can still be filled from the CPU - the save slots' screenshots on the title screen. Its
            // full-size data overran the scaled image (only the width was clamped): GPU writes past the allocation,
            // a device loss below 1x after a seed switch.
            if (w > level_w(b, lv)) w = level_w(b, lv);
            if (hh > level_h(b, lv)) hh = level_h(b, lv);
            VkBufferImageCopy rg = {.bufferOffset = s.off, .bufferRowLength = l.wb * fi->bw, .bufferImageHeight = l.hb * fi->bh,
                                    .imageSubresource = {b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect, lv,
                                                         b->depth > 1 ? 0 : layer, 1},
                                    .imageExtent = {w, b->base_type == VK_IMAGE_VIEW_TYPE_1D || b->base_type == VK_IMAGE_VIEW_TYPE_1D_ARRAY ? 1 : hh,
                                                    b->depth > 1 ? d : 1}};
            if (b->aspect == (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) continue;  // packed D/S: no CPU layout
            vk_dl_note_upload((void *)b->image, rg.imageExtent.width, rg.imageExtent.height, rg.imageSubresource.mipLevel);
                vkCmdCopyBufferToImage(cb, s.buf, b->image, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
        }
    }
}

// LARECOMP_TEX_WATCH=<addr hex>[,<addr hex>...]: log uploads, render-target use, copies and samples of the textures
// at those storage addresses ("[texw]"), to find who fills a texture (e.g. the 3D atmosphere LUT at 1061c5e000).
static uint64_t tw_extra[1024], tw_size[1024];   // added at runtime (LARECOMP_TEX_WATCH_FMT, objects.c nvnTextureInitialize; 64 ran out on terrain tiles)
static int tw_nextra;
void tex_watch_add(uint64_t addr, uint64_t size) {
    for (int i = 0; i < tw_nextra; i++) if (tw_extra[i] == addr) return;
    if (tw_nextra < 1024) { tw_size[tw_nextra] = size; tw_extra[tw_nextra++] = addr; }
}
// [a, a+n) overlaps a texture watched by format (memory copies into / out of its storage)
bool tex_watched_range(uint64_t a, uint64_t n) {
    for (int i = 0; i < tw_nextra; i++) if (a < tw_extra[i] + tw_size[i] && tw_extra[i] < a + n) return true;
    return false;
}
bool tex_watched(uint64_t addr) {
    for (int i = 0; i < tw_nextra; i++) if (tw_extra[i] == addr) return true;
    static uint64_t w[8];
    static int n = -1;
    if (n < 0) {
        n = 0;
        const char *e = getenv("LARECOMP_TEX_WATCH");
        while (e && *e && n < 8) {
            char *end;
            w[n++] = strtoull(e, &end, 16);
            e = *end == ',' ? end + 1 : "";
        }
    }
    for (int i = 0; i < n; i++)
        if (w[i] == addr) return true;
    return false;
}

static void tex_watch_note(NvnTexture *t, TexBackend *b, const char *what) {
    static int nlog;
    if (nlog++ >= 3000) return;   // (was 200: a watch by format (many tiles) used it up at load)
    LOG("[texw] %s addr %llx %ux%ux%u f%#x layers %u frame %llu owned %d uploaded %d\n", what, (unsigned long long)t->addr,
        t->width, t->height, t->depth, t->format, b ? b->layers : 0, (unsigned long long)vk_frame_counter(), b ? b->gpu_owned : -1,
        b ? b->uploaded : -1);
}

// LARECOMP_LOG_RT_FIRST=1: the first GPU write (render / copy target) to each (storage, format, size, levels), with the
// storage range, to find which object fills memory that another object later samples ("[rtfirst]"; 2026-10-16: terrain
// normal maps sampled at addresses no watched object ever rendered).
void rt_first_note(NvnTexture *t, const char *how) {
    static int on = -1;
    if (on < 0) on = getenv("LARECOMP_LOG_RT_FIRST") != NULL;
    if (!on || !t->addr) return;
    static uint64_t seen[1 << 16];
    static int nlog;
    uint64_t k = t->addr ^ ((uint64_t)t->format << 48) ^ ((uint64_t)t->width << 32) ^ ((uint64_t)t->height << 20) ^ t->levels;
    if (!k) k = 1;
    uint32_t s = (uint32_t)(k * 0x9E3779B97F4A7C15ull >> 48);
    for (int p = 0; p < 64; p++, s = (s + 1) & 0xffff) {
        if (seen[s] == k) return;
        if (!seen[s]) { seen[s] = k; break; }
    }
    if (nlog++ >= 20000) return;
    LOG("[rtfirst] %s addr %llx +%llx %ux%ux%u f%#x lv %u flags %#x target %u frame %llu obj %p\n", how, (unsigned long long)t->addr,
        (unsigned long long)tegra_total_size(&t->layout), t->width, t->height, t->depth, t->format, t->levels, t->flags, t->target,
        (unsigned long long)vk_frame_counter(), (void *)t);
}

// ---- write-back. An NVN texture is a view of pool memory: on hardware its rendered texels ARE its storage, and the
// CPU may move them. TotK renders each terrain tile's normal map once (260x260 RG8, 9 levels, flags 0x200), finalizes
// the object, moves the storage to a lower address with the CPU (heap compaction) and samples it through a new object
// there (run375: new objects 0x6000-0x4c000 below finalized rendered ones, uploaded ALL ZERO -> the grass vertex shader
// read a flat-zero normal map and the grass vanished ~3 s into gameplay). Our rendering lives only in the VkImage, so a
// rendered texture the CPU could move (not COMPRESSIBLE 0x8 / DISPLAY 0x1: the game's per-frame targets are all 0x8)
// gets its texels copied back into its storage in the guest layout at the end of the execute that rendered it (one
// readback + one GPU wait per execute that has any, when the GPU would have written them on hardware). A target
// rendered again in a later frame is a per-frame target and is not written back again. LARECOMP_VK_WRITEBACK=0: off.
static NvnTexture *g_wb_pending[512];
static int g_wb_npending;
static uint64_t g_wb_n, g_wb_bytes, g_wb_ns, g_wb_flushes;
static void wb_forget(NvnTexture *t) {   // (queued outside an execute, e.g. by a present, then finalized)
    for (int i = 0; i < g_wb_npending; i++)
        if (g_wb_pending[i] == t) { g_wb_pending[i] = g_wb_pending[--g_wb_npending]; i--; }
}
static int wb_on(void) {
    static int on = -1;
    if (on < 0) { const char *e = getenv("LARECOMP_VK_WRITEBACK"); on = !(e && *e == '0'); }
    return on;
}
static bool wb_candidate(NvnTexture *t, TexBackend *b) {
    const FormatInfo *fi = t->layout.fmt;
    if (!t->addr || (t->flags & (0x1 | 0x8)) || t->target != 1 || t->samples > 1) return false;
    if (!b->image || b->samples > 1 || b->astc_bw || b->blk || b->pending_rgba || b->aspect != VK_IMAGE_ASPECT_COLOR_BIT) return false;
    if (b->layers > 1 || b->depth > 1 || (b->scalable && (b->sx != 1 || b->sy != 1))) return false;
    if (!fi || !fi->bpb || fi->bpb == 3 || fi->bw != 1 || fi->bh != 1 || b->format != fmt_of(t->format).vf) return false;
    return tegra_total_size(&t->layout) <= (4u << 20);
}
static void wb_note_render(NvnTexture *t, TexBackend *b) {
    if (!wb_on() || !b || b->wb_state == 3) return;
    uint64_t f1 = vk_frame_counter() + 1;
    if (b->wb_state == 1) return;                                        // queued already in this execute
    if (b->wb_state == 2 && b->wb_frame != f1) { b->wb_state = 3; return; }  // rendered again in a later frame
    if (!wb_candidate(t, b)) { b->wb_state = 3; return; }
    if (g_wb_npending == (int)(sizeof g_wb_pending / sizeof *g_wb_pending)) return;   // (full: not mid-pass; a later render queues it)
    b->wb_state = 1;
    b->wb_frame = f1;
    g_wb_pending[g_wb_npending++] = t;
}
void vk_writeback_flush(void) {
    if (!g_wb_npending) return;
    extern void renderer_flush(void);
    uint64_t t0 = SDL_GetTicksNS();
    uint64_t off[512], need = 0;
    for (int i = 0; i < g_wb_npending; i++) {
        NvnTexture *t = g_wb_pending[i];
        TexBackend *b = t->backend;
        uint32_t levels = t->layout.levels < b->levels ? t->layout.levels : b->levels;
        off[i] = need;
        for (uint32_t lv = 0; lv < levels; lv++) {
            LevelLayout l = tegra_level(&t->layout, lv);
            need += ((uint64_t)l.wb * t->layout.fmt->bpb * l.hb * l.d + 15) & ~15ull;
        }
    }
    vk_end_pass();
    VkCommandBuffer cb = vk_cmd();
    vk_full_barrier(cb);
    VkSpan out = vk_readback(need);
    for (int i = 0; i < g_wb_npending; i++) {
        NvnTexture *t = g_wb_pending[i];
        TexBackend *b = t->backend;
        uint32_t levels = t->layout.levels < b->levels ? t->layout.levels : b->levels;
        uint64_t o = off[i];
        for (uint32_t lv = 0; lv < levels; lv++) {
            LevelLayout l = tegra_level(&t->layout, lv);
            VkBufferImageCopy c = {.bufferOffset = out.off + o, .bufferRowLength = l.wb, .bufferImageHeight = l.hb,
                                   .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, lv, 0, 1},
                                   .imageExtent = {l.w < level_w(b, lv) ? l.w : level_w(b, lv), l.h < level_h(b, lv) ? l.h : level_h(b, lv), 1}};
            vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, out.buf, 1, &c);
            o += ((uint64_t)l.wb * t->layout.fmt->bpb * l.hb * l.d + 15) & ~15ull;
        }
    }
    renderer_flush();
    for (int i = 0; i < g_wb_npending; i++) {
        NvnTexture *t = g_wb_pending[i];
        TexBackend *b = t->backend;
        const FormatInfo *fi = t->layout.fmt;
        uint32_t levels = t->layout.levels < b->levels ? t->layout.levels : b->levels;
        bool linear = nvn_flags_linear(t->flags);
        uint64_t o = off[i];
        for (uint32_t lv = 0; lv < levels; lv++) {
            LevelLayout l = tegra_level(&t->layout, lv);
            const uint8_t *src = (const uint8_t *)out.ptr + o;
            uint8_t *dst = (uint8_t *)(uintptr_t)t->addr + l.offset;
            uint32_t row = l.wb * fi->bpb;
            if (linear) for (uint32_t y = 0; y < l.hb; y++) memcpy(dst + (uint64_t)y * t->stride, src + (uint64_t)y * row, row);
            else tegra_swizzle(src, dst, l.wb, l.hb, l.d, fi->bpb, l.block_height_log2, l.block_depth_log2);
            o += ((uint64_t)row * l.hb * l.d + 15) & ~15ull;
        }
        b->wb_state = 2;
        g_wb_bytes += o - off[i];
        if (tex_watched(t->addr) || g_wb_n < 4 || !(g_wb_flushes % 64))
            LOG("[wb] %p addr %llx %ux%u f%#x flags %#x lv %u: %llu bytes written back (frame %llu)\n", (void *)t, (unsigned long long)t->addr,
                t->width, t->height, t->format, t->flags, levels, (unsigned long long)(o - off[i]), (unsigned long long)vk_frame_counter());
        g_wb_n++;
    }
    g_wb_ns += SDL_GetTicksNS() - t0;
    g_wb_flushes++;
    if (g_wb_flushes <= 8 || !(g_wb_flushes % 64))
        LOG("[wb] flush %llu: %d textures, %llu KB (total %llu textures, %llu MB, %.1f ms)\n", (unsigned long long)g_wb_flushes,
            g_wb_npending, (unsigned long long)(need >> 10), (unsigned long long)g_wb_n, (unsigned long long)(g_wb_bytes >> 20), g_wb_ns / 1e6);
    g_wb_npending = 0;
}

TexBackend *vk_tex(NvnTexture *t, bool for_render) {
    if (for_render) rt_first_note(t, "render");
    if (tex_watched(t->addr)) {   // every render; each sampled object's first sample (was: the first 20 samples of the run)
        static NvnTexture *seen[4096];
        static int nseen;
        bool first = !for_render;
        for (int i = 0; first && i < nseen; i++) if (seen[i] == t) first = false;
        if (first && nseen < 4096) seen[nseen++] = t;
        if (for_render || first) tex_watch_note(t, tex_backend(t), for_render ? "render" : "sample");
    }
    if (!for_render) {   // sampled / copy source: another object's rendering at the same storage wins
        NvnTexture *w = alias_writer(t);
        if (w) return tex_backend(w);
    }
    TexBackend *b = tex_backend(t);
    if (b->pending_rgba) upload_pending(b);
    if (for_render) { b->gpu_owned = true; alias_note_writer(t); wb_note_render(t, b); }
    else if (!b->gpu_owned) {
        extern uint64_t prof_up_ns;
        uint64_t t0 = prof_up_ns != ~0ull ? SDL_GetTicksNS() : 0;
        upload_texture(t, b);
        if (prof_up_ns != ~0ull) prof_up_ns += SDL_GetTicksNS() - t0;
    }
    return b;
}

void renderer_mark_gpu_owned(NvnTexture *t) { tex_backend(t)->gpu_owned = true; }
bool renderer_is_gpu_owned(NvnTexture *t) { return tex_backend(t)->gpu_owned; }
void backend_pool_written(uint64_t addr, uint64_t size) { (void)addr; (void)size; }
// nvnTextureWriteTexels (objects.c): the next use re-hashes the whole texture (a small region can miss the sampled
// hash, and unchanged textures are only checked every 8th frame).
void backend_texture_cpu_written(NvnTexture *t) { if (t->backend) ((TexBackend *)t->backend)->cpu_dirty = true; }

// Host-produced texel data (decoded video, any thread): applied on the render thread at the texture's next use.
void renderer_upload_rgba(NvnTexture *t, const void *rgba, uint32_t w, uint32_t h, uint32_t stride) {
    TexBackend *b = t->backend;
    if (!b) return;  // the game binds the texture before decoding into it; first frames may arrive earlier
    void *copy = malloc((size_t)stride * h);
    memcpy(copy, rgba, (size_t)stride * h);
    pthread_mutex_lock(&pending_lock);
    free(b->pending_rgba);
    b->pending_rgba = copy;
    b->pending_w = w; b->pending_h = h; b->pending_stride = stride;
    b->gpu_owned = true;
    pthread_mutex_unlock(&pending_lock);
}

// ================================================================== views
#define VIEW_TAB 16384
typedef struct { TexBackend *b; uint64_t k1, k2; VkImageView view; } ViewEnt;
static ViewEnt view_tab[VIEW_TAB];

// Views may still be referenced by submitted or pending work: destroyed once that has completed.
void vk_views_forget(TexBackend *b) { views_forget(b); }
uint32_t vk_view_gen;  // bumped whenever a cached view may be destroyed (exec.c's per-frame texture memo)
static void views_forget(TexBackend *b) {
    vk_view_gen++;
    for (int i = 0; i < VIEW_TAB; i++)
        if (view_tab[i].b == b) { defer((Dead){.view = view_tab[i].view}); view_tab[i].b = NULL; view_tab[i].view = VK_NULL_HANDLE; }
}

VkImageView vk_image_view(TexBackend *b, VkImageViewType type, VkFormat fmt, uint32_t lv0, uint32_t nlv, uint32_t ly0,
                          uint32_t nly, VkComponentMapping sw, VkImageAspectFlags aspect) {
    uint64_t k1 = (uint64_t)type | (uint64_t)fmt << 8 | (uint64_t)lv0 << 40 | (uint64_t)nlv << 48 | (uint64_t)aspect << 56;
    uint64_t k2 = (uint64_t)ly0 | (uint64_t)nly << 16 | (uint64_t)sw.r << 32 | (uint64_t)sw.g << 40 | (uint64_t)sw.b << 48 | (uint64_t)sw.a << 56;
    uint64_t h = ((uint64_t)(uintptr_t)b * 0x9E3779B97F4A7C15ull) ^ (k1 * 0xC2B2AE3D27D4EB4Full) ^ (k2 * 0x165667B19E3779F9ull);
    uint32_t slot = (uint32_t)(h ^ (h >> 32)) & (VIEW_TAB - 1);
    for (int probe = 0; probe < 16; probe++, slot = (slot + 1) & (VIEW_TAB - 1)) {
        ViewEnt *e = &view_tab[slot];
        if (e->b == b && e->k1 == k1 && e->k2 == k2) return e->view;
        if (!e->b) break;
    }
    extern uint64_t prof_views_created;
    prof_views_created++;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = b->image, .viewType = type,
                                .format = fmt, .components = sw, .subresourceRange = {aspect, lv0, nlv, ly0, nly}};
    VkImageView v;
    if (vkCreateImageView(g_vk.dev, &vi, NULL, &v) != VK_SUCCESS) {
        static int nfail;
        if (nfail++ < 20) LOG("[vk] view failed: type %d fmt %d (image fmt %d) levels %u+%u layers %u+%u\n", type, fmt, b->format, lv0, nlv, ly0, nly);
        return VK_NULL_HANDLE;
    }
    ViewEnt *e = &view_tab[slot];
    if (e->b) { defer((Dead){.view = e->view}); vk_view_gen++; }  // evict (16 probes taken)
    *e = (ViewEnt){b, k1, k2, v};
    return v;
}

static VkImageViewType view_type_for(int textype) {
    switch (textype) {
    case TEXTYPE_1D: return VK_IMAGE_VIEW_TYPE_1D;
    case TEXTYPE_3D: return VK_IMAGE_VIEW_TYPE_3D;
    case TEXTYPE_CUBE: return VK_IMAGE_VIEW_TYPE_CUBE;
    case TEXTYPE_2D_ARRAY: return VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    case TEXTYPE_CUBE_ARRAY: return VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    case TEXTYPE_1D_ARRAY: return VK_IMAGE_VIEW_TYPE_1D_ARRAY;
    default: return VK_IMAGE_VIEW_TYPE_2D;
    }
}

static const VkComponentMapping IDENTITY = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                                            VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};

// render: attachment view (one level, one layer). Otherwise a sampled view of type want_textype (TEXTYPE_*),
// with the NVN component swizzle, converting between compatible types the way renderer.m does.
VkImageView vk_view(NvnTexture *t, const NvnTextureView *v, bool has_view, bool render, int want_textype) {
    TexBackend *b = vk_tex(t, render);
    if (!has_view) v = NULL;
    VkFormat fmt = b->format;
    bool color = !(b->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT));
    if (b != t->backend && color && vk_format(t->format) != fmt &&
        format_supports(vk_format(t->format), VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
        fmt = vk_format(t->format);   // another object's image (alias, same texel size): read it in this object's format
    if (v && (v->setmask & 4) && color) {
        VkFormat vf = fmt_of(v->format).vf;
        if (vk_format(t->format) == fmt) fmt = vf;  // (fallback images keep their own format)
    }
    uint32_t lv0 = v && (v->setmask & 1) ? v->base_level : 0;
    uint32_t nlv = v && (v->setmask & 1) && v->num_levels ? v->num_levels : b->levels - (lv0 < b->levels ? lv0 : 0);
    uint32_t ly0 = v && (v->setmask & 2) ? v->min_layer : 0;
    uint32_t nly = v && (v->setmask & 2) && v->num_layers ? v->num_layers : b->layers - (ly0 < b->layers ? ly0 : 0);
    if (lv0 >= b->levels) lv0 = b->levels - 1;
    if (lv0 + nlv > b->levels) nlv = b->levels - lv0;
    if (ly0 >= b->layers) ly0 = b->layers - 1;
    if (ly0 + nly > b->layers) nly = b->layers - ly0;
    if (render) {
        // An image created without attachment usage (block-compressed formats: BC4 64^3 volumes the game fills by
        // rendering encoded blocks through an uncompressed view, run287) must never be rendered to: such a view is invalid
        // and the GPU wrote past the image (run287: nvlddmkm 153, "address write invalid"). Dropped, logged per texture.
        VkImageUsageFlags att = color ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
        if (color && is_bc(b->format) && v && (v->setmask & 4)) {  // encoded blocks through an uncompressed view
            TexBackend *s = block_shadow(t, b, fmt);
            if (s) b = s;   // (same levels / slices / layers; extents in blocks)
            if (s && lv0 >= s->levels) return VK_NULL_HANDLE;  // (2D: the shadow has fewer levels than the BC chain)
        }
        bool slice3d = b->base_type == VK_IMAGE_VIEW_TYPE_3D;
        if (!(b->usage & att) || (slice3d && !(b->cflags & VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT))) {
            static uint64_t seen[64];
            static int nseen;
            bool fresh = true;
            for (int i = 0; i < nseen && fresh; i++) fresh = seen[i] != t->addr;
            if (fresh && nseen < 64) {
                seen[nseen++] = t->addr;
                LOG("[vk] render target not renderable, dropped: %ux%ux%u nvn fmt %#x (view fmt %#x) vk fmt %d usage %#x flags %#x addr %llx\n",
                    t->width, t->height, t->depth, t->format, v && (v->setmask & 4) ? v->format : 0, b->format, b->usage, b->cflags,
                    (unsigned long long)t->addr);
            }
            return VK_NULL_HANDLE;
        }
        VkImageViewType type = b->base_type == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
        if (b->base_type == VK_IMAGE_VIEW_TYPE_3D) {
            // 3D render target (run285: TotK renders level 1+ of a 32x32x32 R8 volume at load; they were dropped, so the
            // volume's mips stayed empty). One depth slice of one level through a 2D view (the image is created
            // 2D_ARRAY_COMPATIBLE); the slice is the NVN view's min layer (the clamp above zeroes it: b->layers is 1).
            // Layered rendering into all slices at once (VS-written layer) is not supported: slice 0 only, logged.
            // LARECOMP_NO_3D_RT=1: not rendered (the old behaviour).
            uint32_t slices = t->depth >> lv0 ? t->depth >> lv0 : 1;
            uint32_t z = v && (v->setmask & 2) ? v->min_layer : 0;
            uint32_t nz = v && (v->setmask & 2) && v->num_layers ? v->num_layers : slices;
            static int nseen, off = -1;
            static uint64_t seen[512];
            if (off < 0) off = getenv("LARECOMP_NO_3D_RT") != NULL;
            uint64_t key = t->addr ^ ((uint64_t)lv0 << 56) ^ ((uint64_t)z << 44);
            bool fresh = true;  // one line per distinct volume / level / slice
            for (int i = 0; i < nseen && fresh; i++) fresh = seen[i] != key;
            if (fresh && nseen < 512) seen[nseen++] = key;
            if (fresh && nseen < 512)
                LOG("[vk] 3D render target %ux%ux%u fmt %#x addr %llx level %u slice %u (+%u of %u)%s\n", t->width, t->height, t->depth,
                    t->format, (unsigned long long)t->addr, lv0, z, nz, slices, off ? ": NOT rendered (LARECOMP_NO_3D_RT)" : "");
            if (off || z >= slices) return VK_NULL_HANDLE;
            return vk_image_view(b, VK_IMAGE_VIEW_TYPE_2D, fmt, lv0, 1, z, 1, IDENTITY, b->aspect);
        }
        return vk_image_view(b, type, fmt, lv0, 1, ly0, 1, IDENTITY, b->aspect);
    }
    VkImageAspectFlags aspect = b->aspect & VK_IMAGE_ASPECT_DEPTH_BIT ? VK_IMAGE_ASPECT_DEPTH_BIT : b->aspect;
    VkComponentMapping sw = IDENTITY;
    if (!(b->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) && b->samples == 1) {
        const uint8_t *s = v && (v->setmask & 8) ? v->swizzle : t->swizzle;
        uint8_t c[4];
        for (int i = 0; i < 4; i++) c[i] = s[i] <= 5 ? s[i] : (uint8_t)(2 + i);
        if (c[0] || c[1] || c[2] || c[3]) {  // never set: identity
            if (fmt_of(t->format).alpha_one)
                for (int i = 0; i < 4; i++) if (c[i] == 5) c[i] = 1;
            // NVN swizzle 0 zero, 1 one, 2..5 r,g,b,a == VkComponentSwizzle - 1
            sw = (VkComponentMapping){(VkComponentSwizzle)(c[0] + 1), (VkComponentSwizzle)(c[1] + 1), (VkComponentSwizzle)(c[2] + 1),
                                      (VkComponentSwizzle)(c[3] + 1)};
        } else if (fmt_of(t->format).alpha_one) {
            sw.a = VK_COMPONENT_SWIZZLE_ONE;
        }
    }
    VkImageViewType want = view_type_for(want_textype);
    VkImageViewType have = b->base_type;
    if (b->samples > 1) return vk_dummy_view(want_textype);  // multisampled images can't be sampled as 2D
    if (want == have) {
        if (have == VK_IMAGE_VIEW_TYPE_CUBE) nly = 6;
        return vk_image_view(b, want, fmt, lv0, nlv, ly0, nly, sw, aspect);
    }
    if (want == VK_IMAGE_VIEW_TYPE_2D && (have == VK_IMAGE_VIEW_TYPE_2D_ARRAY || have == VK_IMAGE_VIEW_TYPE_CUBE))
        return vk_image_view(b, want, fmt, lv0, nlv, ly0, 1, sw, aspect);
    if (want == VK_IMAGE_VIEW_TYPE_2D_ARRAY && (have == VK_IMAGE_VIEW_TYPE_2D || have == VK_IMAGE_VIEW_TYPE_CUBE))
        return vk_image_view(b, want, fmt, lv0, nlv, ly0, have == VK_IMAGE_VIEW_TYPE_CUBE ? nly : 1, sw, aspect);
    if (want == VK_IMAGE_VIEW_TYPE_CUBE && have == VK_IMAGE_VIEW_TYPE_2D_ARRAY && b->layers >= 6 && b->width == b->height)
        return vk_image_view(b, want, fmt, lv0, nlv, ly0 + 6 <= b->layers ? ly0 : 0, 6, sw, aspect);
    if (want == VK_IMAGE_VIEW_TYPE_1D_ARRAY && have == VK_IMAGE_VIEW_TYPE_1D)
        return vk_image_view(b, want, fmt, lv0, nlv, 0, 1, sw, aspect);
    static int nlog;
    if (nlog++ < 30) LOG("[tex] type mismatch: want %d have %d (%ux%ux%u target %u fmt %#x layers %u)\n", want, have, t->width,
                         t->height, t->depth, t->target, t->format, b->layers);
    return vk_dummy_view(want_textype);
}

// 1x1 black textures for unbound slots, per type
VkImageView vk_dummy_view(int textype) {
    static TexBackend dummies[10];
    static VkImageView views[10];
    if (textype < 0 || textype >= 10) textype = TEXTYPE_2D;
    if (views[textype]) return views[textype];
    TexBackend *b = &dummies[textype];
    VkImageViewType vt = view_type_for(textype);
    bool cube = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
                            .imageType = vt == VK_IMAGE_VIEW_TYPE_3D ? VK_IMAGE_TYPE_3D
                                       : vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_1D_ARRAY ? VK_IMAGE_TYPE_1D : VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = cube ? 6 : 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    VK_CHECK(vmaCreateImage(g_vk.vma, &ci, &ai, &b->image, &b->mem, NULL));
    VkCommandBuffer cb = vk_upload_cmd();
    vk_image_init_layout(cb, b->image, VK_IMAGE_ASPECT_COLOR_BIT);
    VkClearColorValue zero = {{0, 0, 0, 0}};
    VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, cube ? 6 : 1};
    vkCmdClearColorImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
    b->format = VK_FORMAT_R8G8B8A8_UNORM;
    b->levels = 1;
    b->layers = cube ? 6 : 1;
    views[textype] = vk_image_view(b, vt, b->format, 0, 1, 0, cube ? 6 : 1, IDENTITY, VK_IMAGE_ASPECT_COLOR_BIT);
    return views[textype];
}

// ================================================================== samplers
static VkCompareOp compare_op(uint32_t f) {
    switch (f) {
    case 1: return VK_COMPARE_OP_NEVER;
    case 2: return VK_COMPARE_OP_LESS;
    case 3: return VK_COMPARE_OP_EQUAL;
    case 4: return VK_COMPARE_OP_LESS_OR_EQUAL;
    case 5: return VK_COMPARE_OP_GREATER;
    case 6: return VK_COMPARE_OP_NOT_EQUAL;
    case 7: return VK_COMPARE_OP_GREATER_OR_EQUAL;
    default: return VK_COMPARE_OP_ALWAYS;
    }
}

VkSampler vk_sampler(NvnSamplerDesc *d) {
    if (d->backend) return (VkSampler)d->backend;
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = d->mag_filter == 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    uint32_t mf = d->min_filter;
    si.minFilter = (mf == 1 || mf == 3 || mf == 5) ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
    si.mipmapMode = mf == 4 || mf == 5 ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    VkSamplerAddressMode modes[3];
    bool border = false;
    for (int i = 0; i < 3; i++) {
        switch (d->wrap[i]) {
        case 0: case 7: modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE; break;
        case 1: modes[i] = VK_SAMPLER_ADDRESS_MODE_REPEAT; break;
        case 2: case 3: modes[i] = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE; break;
        case 4: case 5: modes[i] = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER; border = true; break;
        case 6: modes[i] = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT; break;
        default: modes[i] = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        }
    }
    si.addressModeU = modes[0]; si.addressModeV = modes[1]; si.addressModeW = modes[2];
    si.minLod = d->lod_min;
    si.maxLod = d->lod_max > 1000 ? VK_LOD_CLAMP_NONE : d->lod_max;
    if (mf <= 1) si.maxLod = si.minLod + 0.25f;  // not mipmapped: stay on the base level
    if (si.maxLod < si.minLod) si.maxLod = si.minLod;
    si.mipLodBias = d->lod_bias;
    if (d->max_aniso > 1) { si.anisotropyEnable = VK_TRUE; si.maxAnisotropy = fminf(d->max_aniso, 16); }
    static float lodmin = -1;
    if (lodmin < 0) lodmin = getenv("LARECOMP_LOD_MIN") ? (float)atof(getenv("LARECOMP_LOD_MIN")) : 0;
    if (lodmin > 0 && mf > 1) si.minLod = fmaxf(si.minLod, lodmin);
    static int stest = -1;
    if (stest < 0) stest = getenv("LARECOMP_SAMPLER_TEST") != NULL;
    if (stest && mf > 1) { si.minLod = 0; si.maxLod = VK_LOD_CLAMP_NONE; si.anisotropyEnable = VK_TRUE; si.maxAnisotropy = 16; }
    if (d->compare_mode) {
        si.compareEnable = VK_TRUE;
        si.compareOp = getenv("LARECOMP_NO_SHADOW") ? VK_COMPARE_OP_ALWAYS : compare_op(d->compare_func);
    }
    VkSamplerCustomBorderColorCreateInfoEXT cb = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT};
    if (border) {
        const float *c = d->border;
        if (c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 0) si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
        else if (c[0] == 0 && c[1] == 0 && c[2] == 0 && c[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
        else if (c[0] == 1 && c[1] == 1 && c[2] == 1 && c[3] == 1) si.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
        else if (g_vk.custom_border) {
            memcpy(cb.customBorderColor.float32, c, 16);
            si.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
            si.pNext = &cb;
        } else si.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    }
    VkSampler s;
    VK_CHECK(vkCreateSampler(g_vk.dev, &si, NULL, &s));
    d->backend = (void *)s;
    {   // measurement (blurry ground, 2026-10-14): the guest's LOD settings per sampler, first 64
        static int nlog;
        if (nlog++ < 64)
            LOG("[smp] %p min %u mag %u wrap %u/%u/%u lod %.3f..%.3f bias %.3f aniso %.0f cmp %u -> vk minLod %.3f maxLod %.3f\n",
                (void *)d, d->min_filter, d->mag_filter, d->wrap[0], d->wrap[1], d->wrap[2], d->lod_min, d->lod_max, d->lod_bias,
                (double)d->max_aniso, d->compare_mode, si.minLod, si.maxLod);
    }
    return s;
}

VkSampler vk_default_sampler(bool linear) {
    static VkSampler s[2];
    if (!s[linear]) {
        VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                  .magFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
                                  .minFilter = linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
                                  .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                                  .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.25f};
        VK_CHECK(vkCreateSampler(g_vk.dev, &si, NULL, &s[linear]));
    }
    return s[linear];
}
