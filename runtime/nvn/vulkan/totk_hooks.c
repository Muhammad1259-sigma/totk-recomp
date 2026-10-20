// TotK side of the hooks LASRecomp's Vulkan backend calls into its Link's Awakening runtime (widescreen, pots,
// debug tools, SMAA tables, output scaling). None of those features exist here: neutral answers that leave the
// game's own image untouched.
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "../output_scale.h"

uint64_t plat_now_ns(void) {    // monotonic ns (was in backend_null.c)
    static LARGE_INTEGER f;
    LARGE_INTEGER t;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return (uint64_t)((double)t.QuadPart * 1e9 / (double)f.QuadPart);
}

// Image letterboxed into the window at 16:9. Render targets at the game's sizes, or scaled by TOTK_RES_SCALE
// (textures.c tex_scalable: 16:9 colour/depth targets; 1.3333 = 1600x900 -> 2133x1200, 1080p output -> 1440p;
// "auto" = window height / 1080). Shaders see guest sizes through the rscale table (translate.c).
static double res_scale(double dh) {
    static double s = -1;
    if (s < 0) {
        const char *e = getenv("TOTK_RES_SCALE");
        s = !e || !*e ? 1.0 : !strcmp(e, "auto") ? 0.0 : atof(e);
        if (s != 0.0 && (s < 0.5 || s > 4.0)) s = 1.0;
    }
    if (s == 0.0) { double a = dh / 1080.0; return a < 1.0 ? 1.0 : a > 4.0 ? 4.0 : a; }
    return s;
}

void output_scale(double dw, double dh, bool reshape_allowed, double now, double gpu_ms, OutScale *o) {
    (void)reshape_allowed; (void)now; (void)gpu_ms;
    o->sx = o->sy = res_scale(dh); o->shape = 1.0;
    double w = dw, h = dw * 9.0 / 16.0;
    if (h > dh) { h = dh; w = dh * 16.0 / 9.0; }
    o->vw = w; o->vh = h; o->vx = (dw - w) / 2; o->vy = (dh - h) / 2;
}
double output_scale_dynamic_factor(void) { return 1.0; }

void widescreen_frame_end(uint64_t frame) { (void)frame; }
void widescreen_set_factor(float f) { (void)f; }
bool widescreen_allowed(void) { return false; }
void widescreen_present_crop(uint64_t frame, double shape, float *crop, double *vx, double *vy, double *vw, double *vh) {
    (void)frame; (void)shape; (void)vx; (void)vy; (void)vw; (void)vh;
    crop[0] = 0; crop[1] = 0; crop[2] = 1; crop[3] = 1;    // the whole image
}
void widescreen_backdrop_seen(uint64_t hash) { (void)hash; }
void widescreen_scene_ubo(const float *u, uint64_t size) { (void)u; (void)size; }

void fnprof_frame(uint64_t frame) { (void)frame; }
void warp_frame(uint64_t frame) { (void)frame; }
void reboot_test_frame(uint64_t frame) { (void)frame; }
void save_editor_frame(uint64_t frame) { (void)frame; }
volatile int g_debug_frame_request;

bool pot_level_active(void) { return false; }
bool pot_glow_active(void) { return false; }
bool pot_tint(const float *m, float tint[8], uint64_t mesh) { (void)m; (void)tint; (void)mesh; return false; }
bool pets_hide_item_cursor(void) { return false; }
void escape_bottle_tint(int kind, const float *ui, uint32_t ui_size, float tint[8]) { (void)kind; (void)ui; (void)ui_size; (void)tint; }

// SMAA 1x on the presented image (TOTK_AA=smaa, user 2026-10-17: "make it look less sharp with anti aliasing"):
// Jimenez et al., MIT (runtime/nvn/smaa/LICENSE.txt), files from LASRecomp's third_party/smaa. The reference shader
// source as a C string (tools/smaa_inc.py -> smaa/smaa_hlsl.inc) and its two lookup textures.
#include "../smaa/AreaTex.h"    // 160 x 560, R8G8
#include "../smaa/SearchTex.h"  // 64 x 16, R8
const char smaa_hlsl_src[] =
#include "../smaa/smaa_hlsl.inc"
    ;
const unsigned char *smaa_area_tex(void) { return areaTexBytes; }
const unsigned char *smaa_search_tex(void) { return searchTexBytes; }

float widescreen_effect_stretch(uint64_t fs_hash) { (void)fs_hash; return 1.0f; }
bool pot_instance_gold(const float *m) { (void)m; return false; }
void pot_gold_tint(float tint[8]) { (void)tint; }
