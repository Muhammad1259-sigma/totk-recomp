// Vulkan device, window, swapchain and presentation (counterpart of the window/present half of metal.m).
#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "../shim/las.h"  // TotK: was ../../paths.h
#include "../shim/las.h"  // TotK: was ../../settings.h
#include "../shim/las.h"  // TotK: was ../../ui/menu.h
#include "vk.h"
#include "../output_scale.h"
static void *pcache_saver(void *arg);  // (pipeline cache saver thread, below vk_pipeline_cache_save)
// Hang watch (Linux; first Deck run 2026-10-16: rendering stopped at 274 s without a GPU-hang report and the stuck process
// later kept the Deck from suspending). The present path stamps every frame; the last blocking Vulkan call is noted.
static _Atomic uint64_t g_last_frame_ns;
const char *volatile g_rt_op;      // last blocking Vulkan call started (exec.c / here), NULL when it returned
volatile uint64_t g_rt_op_ns;
static volatile int g_present_tid;
#ifndef _WIN32
#include <sys/syscall.h>
#include "../../linux/plat_posix.h"
static void *hang_watch(void *arg);
#endif
// Pipelines made with dynamic uniform buffers (exec.c make_dynamic_ubos, default) never match entries made with plain
// ones: their own file, so the old 3.2 GB one is not loaded for nothing (run385: loading it + every pipeline new =
// 20 GB peak private, watchdog kill at 1.1 GB free). LARECOMP_VK_DYN_UBO=0 keeps using cache/vk_pipelines.bin.
static const char *pcache_file(void) {
    const char *e = getenv("LARECOMP_VK_DYN_UBO");
    return e && !strcmp(e, "0") ? "cache/vk_pipelines.bin" : "cache/vk_pipelines_dynubo.bin";
}
static size_t pcache_saved;  // size of the last file written / loaded (an unchanged 1.8 GB cache was rewritten every boot)
size_t g_vk_pcache_loaded;   // bytes of cache/vk_pipelines.bin the driver accepted (0: started empty; exec.c pa_warm_load)
extern uint32_t vk_archive_warm_snapshot(void);
extern void vk_archive_warm_commit(uint32_t n);
#include "../shim/las.h"  // TotK: was ../renderer.h

VkG g_vk;
static SDL_Window *window;
static atomic_bool ui_ready;
static VkDebugUtilsMessengerEXT messenger;

// ================================================================== instance / device
static VKAPI_ATTR VkBool32 VKAPI_CALL debug_cb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT type,
                                               const VkDebugUtilsMessengerCallbackDataEXT *d, void *user) {
    (void)type, (void)user;
    // first 3 of each message ID (a few early recurring VUIDs filled the old 500-line cap long before gameplay)
    static int32_t ids[1024];
    static uint8_t cnt[1024];
    static int n;
    if (sev < VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT || n >= 5000) return VK_FALSE;
    uint32_t k = (uint32_t)d->messageIdNumber * 2654435761u >> 22;
    while (cnt[k] && ids[k] != d->messageIdNumber) k = (k + 1) & 1023;
    ids[k] = d->messageIdNumber;
    if (cnt[k] < 255 && cnt[k]++ < 3) { n++; LOG("[vk] %s\n", d->pMessage); }
    return VK_FALSE;
}

// LARECOMP_VK_VALIDATION=1: Khronos validation layer (MSYS2 package vulkan-validation-layers)
static void create_instance(void) {
    Uint32 n_sdl = 0;
    const char *const *sdl_ext = SDL_Vulkan_GetInstanceExtensions(&n_sdl);
    const char *ext[16];
    uint32_t n_ext = 0;
    for (Uint32 i = 0; i < n_sdl && n_ext < 14; i++) ext[n_ext++] = sdl_ext[i];
    bool validate = getenv("LARECOMP_VK_VALIDATION") != NULL;
    if (validate) ext[n_ext++] = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
    const char *layers[] = {"VK_LAYER_KHRONOS_validation"};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "LARecomp",
                             .pEngineName = "LARecomp", .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
                               .enabledExtensionCount = n_ext, .ppEnabledExtensionNames = ext,
                               .enabledLayerCount = validate ? 1 : 0, .ppEnabledLayerNames = layers};
    VkResult r = vkCreateInstance(&ci, NULL, &g_vk.inst);
    if (r != VK_SUCCESS && validate) {
        LOG("[vk] validation layer unavailable (%d); continuing without\n", r);
        ci.enabledLayerCount = 0;
        ci.enabledExtensionCount--;
        validate = false;
        r = vkCreateInstance(&ci, NULL, &g_vk.inst);
    }
    if (r != VK_SUCCESS) FATAL("vkCreateInstance: %d (is a Vulkan 1.3 driver installed?)\n", r);
    if (validate) {
        PFN_vkCreateDebugUtilsMessengerEXT f =
            (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(g_vk.inst, "vkCreateDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT mi = {
            .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
            .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
            .pfnUserCallback = debug_cb};
        if (f) f(g_vk.inst, &mi, NULL, &messenger);
    }
}

static bool has_ext(VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++) if (!strcmp(e[i].extensionName, name)) return true;
    return false;
}

// Discrete GPU first (LARECOMP_GPU=<index> picks one), needs Vulkan 1.3 and a queue that can present.
static void pick_device(void) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_vk.inst, &n, NULL);
    if (!n) FATAL("no Vulkan device\n");
    VkPhysicalDevice *pd = calloc(n, sizeof *pd);
    vkEnumeratePhysicalDevices(g_vk.inst, &n, pd);
    int want = getenv("LARECOMP_GPU") ? atoi(getenv("LARECOMP_GPU")) : -1, best = -1, best_score = -1;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd[i], &p);
        LOG("[vk] gpu %u: %s (api %u.%u.%u)\n", i, p.deviceName, VK_API_VERSION_MAJOR(p.apiVersion),
            VK_API_VERSION_MINOR(p.apiVersion), VK_API_VERSION_PATCH(p.apiVersion));
        if (p.apiVersion < VK_API_VERSION_1_3) continue;
        int score = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 3 : p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 2 : 1;
        if ((int)i == want) score = 100;
        if (score > best_score) { best_score = score; best = (int)i; }
    }
    if (best < 0) FATAL("no Vulkan 1.3 device\n");
    g_vk.phys = pd[best];
    free(pd);
    vkGetPhysicalDeviceProperties(g_vk.phys, &g_vk.props);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq, sizeof *qp);
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys, &nq, qp);
    g_vk.qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(g_vk.phys, i, g_vk.surface, &present);
        if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && present) { g_vk.qfam = i; break; }
    }
    free(qp);
    if (g_vk.qfam == UINT32_MAX) FATAL("no graphics+present queue\n");
    LOG("[vk] using %s\n", g_vk.props.deviceName);
}

static bool g_fault_ext, g_fault_binary;
static void fault_info(void);
// The launcher's "> play.log" empties the log at the next launch: keep a copy of a crashed session's log as
// crash_<unix time>.log next to the driver dump (user 2026-10-19: hangs are random, relaunching lost the 16:10 one)
static void save_log_copy(void) {
#ifdef _WIN32
    fflush(stdout);
    fflush(stderr);
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    wchar_t src[MAX_PATH];
    DWORD n = out && GetFileType(out) == FILE_TYPE_DISK ? GetFinalPathNameByHandleW(out, src, MAX_PATH, 0) : 0;
    if (!n || n >= MAX_PATH) return;
    HANDLE in = CreateFileW(src, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, 0, NULL);
    if (in == INVALID_HANDLE_VALUE) return;
    char name[64];
    snprintf(name, sizeof name, "crash_%lld.log", (long long)time(NULL));
    FILE *fp = fopen(name, "wb");
    char buf[1 << 16];
    DWORD got;
    while (fp && ReadFile(in, buf, sizeof buf, &got, NULL) && got) fwrite(buf, 1, got, fp);
    CloseHandle(in);
    if (fp) { fclose(fp); LOG("[vk] log copied to %s\n", name); }
#endif
}
// After VK_ERROR_DEVICE_LOST: what the driver says faulted (VK_EXT_device_fault), into the log; then the log copy.
void vk_device_fault_report(void) {
    static int once;
    if (once++) return;
    if (g_fault_ext) fault_info();
    save_log_copy();
}
static void fault_info(void) {
    PFN_vkGetDeviceFaultInfoEXT get = (PFN_vkGetDeviceFaultInfoEXT)vkGetDeviceProcAddr(g_vk.dev, "vkGetDeviceFaultInfoEXT");
    if (!get) return;
    VkDeviceFaultCountsEXT c = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_COUNTS_EXT};
    if (get(g_vk.dev, &c, NULL) != VK_SUCCESS) { LOG("[vk] device fault: no info\n"); return; }
    VkDeviceFaultAddressInfoEXT addr[16];
    VkDeviceFaultVendorInfoEXT vend[16];
    if (c.addressInfoCount > 16) c.addressInfoCount = 16;
    if (c.vendorInfoCount > 16) c.vendorInfoCount = 16;
    void *bin = g_fault_binary && c.vendorBinarySize ? malloc(c.vendorBinarySize) : NULL;
    if (!bin) c.vendorBinarySize = 0;
    VkDeviceFaultInfoEXT info = {.sType = VK_STRUCTURE_TYPE_DEVICE_FAULT_INFO_EXT, .pAddressInfos = addr, .pVendorInfos = vend,
                                 .pVendorBinaryData = bin};
    if (get(g_vk.dev, &c, &info) != VK_SUCCESS && 0) return;
    if (bin && c.vendorBinarySize) {   // for the vendor's crash-dump tools (NVIDIA: Nsight Graphics / Aftermath)
        char name[64];
        snprintf(name, sizeof name, "gpu_fault_%lld.bin", (long long)time(NULL));
        FILE *fp = fopen(name, "wb");
        size_t w = fp ? fwrite(bin, 1, c.vendorBinarySize, fp) : 0;
        if (fp) fclose(fp);
        LOG("[vk] device fault: driver crash dump %s (%zu of %llu bytes)\n", name, w, (unsigned long long)c.vendorBinarySize);
    }
    free(bin);
    LOG("[vk] device fault: %s\n", info.description);
    static const char *const kind[] = {"none", "read invalid", "write invalid", "execute invalid", "ip unknown", "ip invalid", "ip valid"};
    for (uint32_t i = 0; i < c.addressInfoCount; i++)
        LOG("[vk]   address %s %#llx (precision %#llx)\n", addr[i].addressType < 7 ? kind[addr[i].addressType] : "?",
            (unsigned long long)addr[i].reportedAddress, (unsigned long long)addr[i].addressPrecision);
    for (uint32_t i = 0; i < c.vendorInfoCount; i++)
        LOG("[vk]   vendor: %s (code %#llx data %#llx)\n", vend[i].description, (unsigned long long)vend[i].vendorFaultCode,
            (unsigned long long)vend[i].vendorFaultData);
}

bool vk_dyn_clamp;   // depth clamp is dynamic state in the GPL pre-rasterization libraries (create_device; exec.c)
static void create_device(void) {
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(g_vk.phys, NULL, &ne, ep);
    const char *ext[24];
    uint32_t n_ext = 0;
    ext[n_ext++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    g_vk.custom_border = has_ext(ep, ne, VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME);
    if (g_vk.custom_border) ext[n_ext++] = VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME;
    g_vk.divisor = has_ext(ep, ne, VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME);
    if (g_vk.divisor) ext[n_ext++] = VK_EXT_VERTEX_ATTRIBUTE_DIVISOR_EXTENSION_NAME;
    g_vk.host_import = has_ext(ep, ne, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);  // guest heap as a GPU buffer (exec.c)
    // LARECOMP_VK_NO_HOST_IMPORT=1: no host-pointer imports (diagnostic A/B on amdgpu, where an import is a userptr buffer;
    // shader global memory then reads zero, so the picture is wrong)
    if (getenv("LARECOMP_VK_NO_HOST_IMPORT")) { g_vk.host_import = false; LOG("[vk] host-pointer imports off (LARECOMP_VK_NO_HOST_IMPORT)\n"); }
    if (g_vk.host_import) ext[n_ext++] = VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME;
    // dma-buf imports (Linux, amdgpu): the guest heap / alias window as driver-owned buffer objects instead of userptr host
    // imports (runtime/linux/plat_posix.cpp rt_gpu_bo_map, exec.c host_import)
    g_vk.dmabuf_import = has_ext(ep, ne, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME) && has_ext(ep, ne, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME);
    if (g_vk.dmabuf_import) {
        ext[n_ext++] = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
        ext[n_ext++] = VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME;
    }
    bool fault = has_ext(ep, ne, VK_EXT_DEVICE_FAULT_EXTENSION_NAME);  // device-lost diagnostics (vk_device_fault_report)
    if (fault) ext[n_ext++] = VK_EXT_DEVICE_FAULT_EXTENSION_NAME;
    g_fault_ext = fault;
    // NVIDIA checkpoints: exec.c marks every draw / dispatch / pass / copy; after a GPU hang the driver reports the last
    // mark each pipeline stage reached (exec.c hang_report). LARECOMP_CHECKPOINTS=0: off.
    g_vk.checkpoints = has_ext(ep, ne, VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME) &&
                       !(getenv("LARECOMP_CHECKPOINTS") && !strcmp(getenv("LARECOMP_CHECKPOINTS"), "0"));
    if (g_vk.checkpoints) ext[n_ext++] = VK_NV_DEVICE_DIAGNOSTIC_CHECKPOINTS_EXTENSION_NAME;
    LOG("[vk] GPU checkpoints (hang reports): %s\n", g_vk.checkpoints ? "on" : "off");
    // shader libraries + fast link (DXVK's stutter fix): each program's shaders compile once into libraries, a new draw
    // state only links them (exec.c). LARECOMP_NO_GPL=1: monolithic pipelines only (old).
    VkPhysicalDeviceDriverProperties drv = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT gplp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_PROPERTIES_EXT, .pNext = &drv};
    VkPhysicalDeviceProperties2 pp2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &gplp};
    vkGetPhysicalDeviceProperties2(g_vk.phys, &pp2);
    // NVIDIA's compiler miscompiles some programs in one shader form: those take the other (translate.c nv_pcloop)
    extern bool maxwell_nvidia;
    maxwell_nvidia = drv.driverID == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
    LOG("[vk] driver %s %s%s\n", drv.driverName, drv.driverInfo, maxwell_nvidia ? " (NVIDIA shader-form list on)" : "");
    g_vk.gpl = has_ext(ep, ne, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME) && has_ext(ep, ne, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) &&
               gplp.graphicsPipelineLibraryFastLinking && !getenv("LARECOMP_NO_GPL");
    if (g_vk.gpl) {
        ext[n_ext++] = VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME;
        ext[n_ext++] = VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME;
    }
    LOG("[vk] graphics pipeline library: %s\n", g_vk.gpl ? "on (fast link)" : "off");
    // Depth clamp as dynamic state of the GPL pre-rasterization library (U21, 2026-10-20: TotK's shadow casters draw with
    // depth clamp; a static clamp kept those keys off the fast link, so their draws were skipped until a monolithic build
    // on a worker finished = shadows "spawn in"). exec.c sets it per draw. LARECOMP_NO_DYN_CLAMP=1: off (old).
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    vk_dyn_clamp = g_vk.gpl && has_ext(ep, ne, VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME) && !getenv("LARECOMP_NO_DYN_CLAMP");
    if (vk_dyn_clamp) {
        VkPhysicalDeviceFeatures2 q = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &eds3};
        vkGetPhysicalDeviceFeatures2(g_vk.phys, &q);
        vk_dyn_clamp = eds3.extendedDynamicState3DepthClampEnable && q.features.depthClamp;
    }
    if (vk_dyn_clamp) ext[n_ext++] = VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME;
    LOG("[vk] dynamic depth clamp (GPL fast link for clamped draws): %s\n", vk_dyn_clamp ? "on" : "off");
    // Attachment feedback loops (2026-10-19 NVIDIA hang, CURRENT.md): TotK's G-buffer decals sample the targets they render
    // to; exec.c declares those loops (feedback-loop image usage, VkAttachmentFeedbackLoopInfoEXT, dynamic state) instead
    // of leaving them undefined. Layouts stay GENERAL (unified image layouts). LARECOMP_NO_FEEDBACK_LOOP_EXT=1: off.
    VkPhysicalDeviceUnifiedImageLayoutsFeaturesKHR uil = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_UNIFIED_IMAGE_LAYOUTS_FEATURES_KHR};
    VkPhysicalDeviceAttachmentFeedbackLoopDynamicStateFeaturesEXT fld = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_FEATURES_EXT, .pNext = &uil};
    VkPhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT fll = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_FEATURES_EXT, .pNext = &fld};
    g_vk.feedback_loop = has_ext(ep, ne, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME) &&
                         has_ext(ep, ne, VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME) &&
                         has_ext(ep, ne, VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME) && !getenv("LARECOMP_NO_FEEDBACK_LOOP_EXT");
    if (g_vk.feedback_loop) {
        VkPhysicalDeviceFeatures2 q = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &fll};
        vkGetPhysicalDeviceFeatures2(g_vk.phys, &q);
        g_vk.feedback_loop = fll.attachmentFeedbackLoopLayout && fld.attachmentFeedbackLoopDynamicState && uil.unifiedImageLayouts;
        uil.unifiedImageLayoutsVideo = VK_FALSE;   // (no video queues)
    }
    if (g_vk.feedback_loop) {
        ext[n_ext++] = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_EXTENSION_NAME;
        ext[n_ext++] = VK_EXT_ATTACHMENT_FEEDBACK_LOOP_DYNAMIC_STATE_EXTENSION_NAME;
        ext[n_ext++] = VK_KHR_UNIFIED_IMAGE_LAYOUTS_EXTENSION_NAME;
    }
    LOG("[vk] attachment feedback loops: %s\n", g_vk.feedback_loop ? "declared (EXT layout + dynamic state, unified layouts)" : "off");
    free(ep);

    // features the backend relies on (all present on current desktop drivers)
    VkPhysicalDeviceFaultFeaturesEXT have_ff = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT};
    VkPhysicalDeviceFeatures2 have = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = fault ? &have_ff : NULL};
    vkGetPhysicalDeviceFeatures2(g_vk.phys, &have);
    VkPhysicalDeviceVertexAttributeDivisorFeaturesEXT div = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_ATTRIBUTE_DIVISOR_FEATURES_EXT,
        .vertexAttributeInstanceRateDivisor = VK_TRUE, .vertexAttributeInstanceRateZeroDivisor = VK_FALSE};
    VkPhysicalDeviceCustomBorderColorFeaturesEXT cbc = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT, .customBorderColors = VK_TRUE,
        .customBorderColorWithoutFormat = VK_TRUE, .pNext = g_vk.divisor ? &div : NULL};
    VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                            .dynamicRendering = VK_TRUE, .synchronization2 = VK_TRUE,
                                            .shaderDemoteToHelperInvocation = VK_TRUE, .shaderTerminateInvocation = VK_TRUE,
                                            .maintenance4 = VK_TRUE,
                                            .pNext = g_vk.custom_border ? (void *)&cbc : g_vk.divisor ? (void *)&div : NULL};
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                            .timelineSemaphore = VK_TRUE, .shaderInt8 = VK_TRUE, .samplerMirrorClampToEdge = VK_TRUE,
                                            .uniformAndStorageBuffer8BitAccess = VK_TRUE, .storageBuffer8BitAccess = VK_TRUE,
                                            .scalarBlockLayout = VK_TRUE, .pNext = &f13};
    VkPhysicalDeviceVulkan11Features f11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
                                            .storageBuffer16BitAccess = VK_TRUE, .pNext = &f12};
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gplf = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT, .graphicsPipelineLibrary = VK_TRUE};
    if (g_vk.gpl) { gplf.pNext = f13.pNext; f13.pNext = &gplf; }
    if (g_vk.feedback_loop) { uil.pNext = f13.pNext; f13.pNext = &fll; }   // (fll -> fld -> uil, all queried true)
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT,
                                                              .extendedDynamicState3DepthClampEnable = VK_TRUE};
    if (vk_dyn_clamp) { eds3f.pNext = f13.pNext; f13.pNext = &eds3f; }
    // vendor binary = the driver's own GPU crash dump (NVIDIA 617: supported), saved by vk_device_fault_report
    VkPhysicalDeviceFaultFeaturesEXT ff = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FAULT_FEATURES_EXT, .deviceFault = VK_TRUE,
                                           .deviceFaultVendorBinary = have_ff.deviceFaultVendorBinary, .pNext = &f11};
    g_fault_binary = have_ff.deviceFaultVendorBinary;
    VkPhysicalDeviceFeatures2 f = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = fault ? (void *)&ff : (void *)&f11};
    VkPhysicalDeviceFeatures *b = &f.features, *h = &have.features;
    b->independentBlend = h->independentBlend;
    b->dualSrcBlend = h->dualSrcBlend;
    b->fillModeNonSolid = h->fillModeNonSolid;
    b->samplerAnisotropy = h->samplerAnisotropy;
    b->textureCompressionBC = h->textureCompressionBC;
    b->shaderInt64 = h->shaderInt64;
    b->shaderFloat64 = h->shaderFloat64;  // DADD/DMUL/DFMA/F2F.F64 ('material' VS with double-precision positions)
    b->depthBiasClamp = h->depthBiasClamp;
    b->depthClamp = h->depthClamp;
    b->largePoints = h->largePoints;
    b->wideLines = h->wideLines;
    b->imageCubeArray = h->imageCubeArray;
    b->sampleRateShading = h->sampleRateShading;
    b->fragmentStoresAndAtomics = h->fragmentStoresAndAtomics;
    b->vertexPipelineStoresAndAtomics = h->vertexPipelineStoresAndAtomics;
    b->robustBufferAccess = h->robustBufferAccess;  // guest shaders index cbufs freely: keep OOB reads harmless
    b->shaderClipDistance = h->shaderClipDistance;
    if (!h->shaderInt64) LOG("[vk] warning: no shaderInt64 (64-bit address math in shaders)\n");
    if (!h->shaderFloat64) LOG("[vk] warning: no shaderFloat64 (double-precision 'material' vertex shaders will not build)\n");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = g_vk.qfam,
                                  .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qi, .enabledExtensionCount = n_ext, .ppEnabledExtensionNames = ext};
    VK_CHECK(vkCreateDevice(g_vk.phys, &di, NULL, &g_vk.dev));
    vkGetDeviceQueue(g_vk.dev, g_vk.qfam, 0, &g_vk.queue);

    VmaAllocatorCreateInfo ai = {.physicalDevice = g_vk.phys, .device = g_vk.dev, .instance = g_vk.inst,
                                 .vulkanApiVersion = VK_API_VERSION_1_3};
    VK_CHECK(vmaCreateAllocator(&ai, &g_vk.vma));

    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    VK_CHECK(vkCreateSemaphore(g_vk.dev, &si, NULL, &g_vk.timeline));
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = g_vk.qfam};
    VK_CHECK(vkCreateCommandPool(g_vk.dev, &pi, NULL, &g_vk.pool));

    // pipeline cache: cache/vk_pipelines.bin (written by vk_pipeline_cache_save)
    size_t sz = 0;
    void *data = NULL;
    extern bool vk_archive_warm_stale(void);   // (exec.c: new shader generation -> the old entries are dead weight)
    extern void vk_archive_warm_reset(void);
    bool stale = vk_archive_warm_stale();
    if (stale) {
        LOG("[vk] pipeline cache: archive warm-up stamp changed: %s not loaded (previous shader generation), starting empty\n", pcache_file());
        remove(pcache_file());     // (dead weight; this session's saves replace it)
        vk_archive_warm_reset();   // so the next launch loads what this session saves
    }
    FILE *fp = stale ? NULL : fopen(pcache_file(), "rb");
    if (fp) {
        // 64-bit size: with the effect shaders warmed the cache passed 2 GB (2.6 GB after run354) and ftell's 32-bit
        // long made every later boot die here (runs 355/356 exit 127 one second in; run357 without the file ran).
#ifdef _WIN32
        _fseeki64(fp, 0, SEEK_END);
        long long end = _ftelli64(fp);
        _fseeki64(fp, 0, SEEK_SET);
#else
        fseeko(fp, 0, SEEK_END);
        long long end = (long long)ftello(fp);
        fseeko(fp, 0, SEEK_SET);
#endif
        sz = end > 0 ? (size_t)end : 0;
        data = sz ? malloc(sz) : NULL;
        if (!data || fread(data, 1, sz, fp) != sz) {
            if (sz) LOG("[vk] pipeline cache: %s (%lld bytes) not read: starting empty\n", pcache_file(), end);
            sz = 0;
        }
        fclose(fp);
    }
    VkPipelineCacheCreateInfo pc = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = sz,
                                    .pInitialData = data};
    if (vkCreatePipelineCache(g_vk.dev, &pc, NULL, &g_vk.pcache) != VK_SUCCESS) {
        pc.initialDataSize = 0;
        VK_CHECK(vkCreatePipelineCache(g_vk.dev, &pc, NULL, &g_vk.pcache));
    }
    LOG("[vk] pipeline cache: %zu KB loaded from %s\n", sz >> 10, pcache_file());
    pcache_saved = sz;
    if (pc.initialDataSize) g_vk_pcache_loaded = sz;
    free(data);
    // The cache used to be written only by quick_exit_handler: test runs end by timeout and the user's window close did
    // not reach it, so cache/vk_pipelines.bin never existed (2026-10-15: only a half-written .tmp) and every launch
    // compiled every pipeline again (run183: 252 compiles, a 287 ms hitch at gameplay start). Saved from a background
    // thread every 10 s when the cache grew (VkPipelineCache is internally synchronized).
    static pthread_t saver;
    pthread_create(&saver, NULL, pcache_saver, NULL);
    pthread_detach(saver);
#ifndef _WIN32
    static pthread_t watch;
    pthread_create(&watch, NULL, hang_watch, NULL);
    pthread_detach(watch);
#endif
}

static pthread_mutex_t pcache_save_lock = PTHREAD_MUTEX_INITIALIZER;

void vk_pipeline_cache_save(void) {
    pthread_mutex_lock(&pcache_save_lock);
    size_t sz = 0;
    uint32_t warm_n = vk_archive_warm_snapshot();   // archive programs warmed so far: inside the data read below
    if (vkGetPipelineCacheData(g_vk.dev, g_vk.pcache, &sz, NULL) != VK_SUCCESS || !sz || sz == pcache_saved) {
        if (sz && sz == pcache_saved && pcache_saved) vk_archive_warm_commit(warm_n);   // unchanged since the file
        pthread_mutex_unlock(&pcache_save_lock);
        return;
    }
    void *data = malloc(sz);
    if (vkGetPipelineCacheData(g_vk.dev, g_vk.pcache, &sz, data) == VK_SUCCESS) {
        mkdir("cache", 0755);
        char tmp[96];
        snprintf(tmp, sizeof tmp, "%s.tmp", pcache_file());
        FILE *f = fopen(tmp, "wb");
        if (f) {
            bool ok = fwrite(data, 1, sz, f) == sz;
            fclose(f);
            if (ok) { remove(pcache_file()); ok = rename(tmp, pcache_file()) == 0; }
            if (ok) { pcache_saved = sz; vk_archive_warm_commit(warm_n); }
            static int nlog;
            if (nlog++ < 8) LOG("[vk] pipeline cache saved: %zu KB%s\n", sz >> 10, ok ? "" : " (FAILED)");
        }
    }
    free(data);
    pthread_mutex_unlock(&pcache_save_lock);
}

static void *pcache_saver(void *arg) {
    (void)arg;
#ifndef _WIN32
    { extern int pthread_setname_np(pthread_t, const char *); pthread_setname_np(pthread_self(), "vk-pcache-save"); }   // ([cpu] line)
#endif
    extern int vk_precompile_progress(int *done, int *total);
    // Every save copies the whole cache into RAM and rewrites the file: 1.8 GB, 2.6 GB with the effect shaders half
    // warmed (run354: peak private 15.5 GB vs 12.4). The user's 21:26 session wrote 1.8 GB 8 times in ~6 min of play
    // (5 of the 8 stats windows after a save under 55 fps vs 30 % overall: weak but suspicious). Now: every 3 min while
    // the precompile runs, once when it ends, then every 2 min only if the cache grew >= 16 MB (any growth: 10 min).
    // The exit handler still saves.
    bool was_running = false;
    int since = 0;
    for (int n = 0;; n++) {
        SDL_Delay(10000);
        int d, t;
        if (vk_precompile_progress(&d, &t)) {
            was_running = true;
            if (n % 18 == 17) vk_pipeline_cache_save();
            continue;
        }
        if (was_running) { was_running = false; vk_pipeline_cache_save(); since = 0; continue; }
        if (++since < 12) continue;
        size_t sz = 0;
        if (vkGetPipelineCacheData(g_vk.dev, g_vk.pcache, &sz, NULL) != VK_SUCCESS) continue;
        if (sz >= pcache_saved + ((size_t)16 << 20) || (sz != pcache_saved && since >= 60)) { vk_pipeline_cache_save(); since = 0; }
    }
    return NULL;
}

// ================================================================== swapchain
static VkPresentModeKHR present_mode(void) {
    // MAILBOX by default: the game paces itself (vi.c's emulated 60 Hz vsync), and with FIFO every submission
    // queued behind the one waiting for a window image, so the game's fence waits followed the presentation engine:
    // a throttled / captured / busy compositor slowed the whole game (4-21 fps with the GPU mostly idle; mailbox:
    // 60). No tearing. LARECOMP_VK_PRESENT=fifo|immediate|mailbox to choose.
    const char *e = getenv("LARECOMP_VK_PRESENT");
    VkPresentModeKHR want = !e || !strcmp(e, "mailbox") ? VK_PRESENT_MODE_MAILBOX_KHR
                          : !strcmp(e, "immediate") ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR;
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.phys, g_vk.surface, &n, NULL);
    VkPresentModeKHR m[16];
    if (n > 16) n = 16;
    vkGetPhysicalDeviceSurfacePresentModesKHR(g_vk.phys, g_vk.surface, &n, m);
    for (uint32_t i = 0; i < n; i++) if (m[i] == want) return want;
    return VK_PRESENT_MODE_FIFO_KHR;
}

static void create_swapchain(void) {
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(g_vk.phys, g_vk.surface, &caps));
    int w, h;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    VkExtent2D ext = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent : (VkExtent2D){(uint32_t)w, (uint32_t)h};
    if (!ext.width || !ext.height) return;  // minimised
    uint32_t nf = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.phys, g_vk.surface, &nf, NULL);
    VkSurfaceFormatKHR fmts[64];
    if (nf > 64) nf = 64;
    vkGetPhysicalDeviceSurfaceFormatsKHR(g_vk.phys, g_vk.surface, &nf, fmts);
    // scanout shows the bytes the game wrote (see the raw views in the present pass): a UNORM swapchain like Metal's
    VkSurfaceFormatKHR sf = fmts[0];
    for (uint32_t i = 0; i < nf; i++)
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM && fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { sf = fmts[i]; break; }
    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;
    if (count > 8) count = 8;
    VkSwapchainKHR old = g_vk.swapchain;
    VkSwapchainCreateInfoKHR ci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = g_vk.surface, .minImageCount = count,
        .imageFormat = sf.format, .imageColorSpace = sf.colorSpace, .imageExtent = ext, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, .presentMode = present_mode(), .clipped = VK_TRUE,
        .oldSwapchain = old};
    VK_CHECK(vkCreateSwapchainKHR(g_vk.dev, &ci, NULL, &g_vk.swapchain));
    for (uint32_t i = 0; i < g_vk.sc_count; i++) vkDestroyImageView(g_vk.dev, g_vk.sc_views[i], NULL);
    if (old) vkDestroySwapchainKHR(g_vk.dev, old, NULL);
    g_vk.sc_format = sf.format;
    g_vk.sc_extent = ext;
    g_vk.sc_count = 8;
    vkGetSwapchainImagesKHR(g_vk.dev, g_vk.swapchain, &g_vk.sc_count, g_vk.sc_images);
    for (uint32_t i = 0; i < g_vk.sc_count; i++) {
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_vk.sc_images[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = sf.format,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VK_CHECK(vkCreateImageView(g_vk.dev, &vi, NULL, &g_vk.sc_views[i]));
    }
}

static void recreate_swapchain(void) {
    vkDeviceWaitIdle(g_vk.dev);
    create_swapchain();
}

// ================================================================== frames (start screen and game)
// Acquire semaphores rotate per frame in flight; a slot is reused once the submission that waited on it is done.
#define FRAMES 3
static VkSemaphore acquire_sem[FRAMES], done_sem[8];
static uint64_t slot_seq[FRAMES];
static int slot;

static VkSemaphore binary_semaphore(void) {
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VkSemaphore s;
    VK_CHECK(vkCreateSemaphore(g_vk.dev, &si, NULL, &s));
    return s;
}

static void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .oldLayout = from, .newLayout = to, .image = img,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
}

typedef struct { uint32_t image; VkSemaphore acquired; bool ok; } Frame;
static Frame cur_frame;

// Acquire the next window image and begin rendering into it (cleared to `clear`). Returns false when there is
// nothing to draw into (minimised window).
bool vk_frame_begin(const float clear[4]) {
    cur_frame.ok = false;
    int w, h;
    SDL_GetWindowSizeInPixels(window, &w, &h);
    if (!w || !h) { atomic_store(&g_last_frame_ns, SDL_GetTicksNS()); return false; }  // minimised (not a hang)
    if (!g_vk.swapchain || (uint32_t)w != g_vk.sc_extent.width || (uint32_t)h != g_vk.sc_extent.height) recreate_swapchain();
    if (!g_vk.swapchain) return false;
    static int trace = -1;
    if (trace < 0) trace = getenv("LARECOMP_VK_TRACE") != NULL;
    if (trace) LOG("[vk] frame begin %dx%d (swapchain %ux%u)\n", w, h, g_vk.sc_extent.width, g_vk.sc_extent.height);
    slot = (slot + 1) % FRAMES;
    if (!acquire_sem[slot]) acquire_sem[slot] = binary_semaphore();
    vk_wait_seq(slot_seq[slot]);
    VkResult r;
    for (uint64_t a0 = SDL_GetTicksNS();;) {   // (UINT64_MAX before: a compositor that stops releasing images = a silent hang)
        g_rt_op = "vkAcquireNextImageKHR", g_rt_op_ns = SDL_GetTicksNS();
        r = vkAcquireNextImageKHR(g_vk.dev, g_vk.swapchain, 2000000000ull, acquire_sem[slot], VK_NULL_HANDLE, &cur_frame.image);
        g_rt_op = NULL;
        if (r != VK_TIMEOUT && r != VK_NOT_READY) break;
        LOG("[hang] no window image from the compositor for %.0f s (vkAcquireNextImageKHR)\n", (SDL_GetTicksNS() - a0) / 1e9);
        fflush(stdout);
    }
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        recreate_swapchain();
        r = vkAcquireNextImageKHR(g_vk.dev, g_vk.swapchain, UINT64_MAX, acquire_sem[slot], VK_NULL_HANDLE, &cur_frame.image);
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) { LOG("[vk] acquire failed: %d\n", r); return false; }
    cur_frame.acquired = acquire_sem[slot];
    slot_seq[slot] = vk_pending_seq();
    VkCommandBuffer cb = vk_cmd();
    image_barrier(cb, g_vk.sc_images[cur_frame.image], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = g_vk.sc_views[cur_frame.image],
                                    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
                                    .clearValue.color.float32 = {clear[0], clear[1], clear[2], clear[3]}};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, g_vk.sc_extent}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkViewport vp = {0, 0, (float)g_vk.sc_extent.width, (float)g_vk.sc_extent.height, 0, 1};
    VkRect2D sc = {{0, 0}, g_vk.sc_extent};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    { extern void vk_dyn_invalidate(void); vk_dyn_invalidate(); }
    cur_frame.ok = true;
    return true;
}

// Passes of their own between vk_frame_begin and vk_frame_end (exec.c post passes): end the window image's
// rendering, and begin it again keeping what was drawn.
void vk_frame_suspend(void) {
    if (cur_frame.ok) vkCmdEndRendering(vk_cmd());
}
void vk_frame_resume(void) {
    if (!cur_frame.ok) return;
    VkCommandBuffer cb = vk_cmd();
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = g_vk.sc_views[cur_frame.image],
                                    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, g_vk.sc_extent}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    VkViewport vp = {0, 0, (float)g_vk.sc_extent.width, (float)g_vk.sc_extent.height, 0, 1};
    VkRect2D sc = {{0, 0}, g_vk.sc_extent};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    { extern void vk_dyn_invalidate(void); vk_dyn_invalidate(); }
}

VkExtent2D vk_frame_extent(void) { return g_vk.sc_extent; }
void *vk_frame_cmd(void) { return (void *)vk_cmd(); }

static void save_window_image(VkCommandBuffer cb, VkSpan *out) {
    uint32_t w = g_vk.sc_extent.width, h = g_vk.sc_extent.height;
    *out = vk_readback((uint64_t)w * h * 4);
    image_barrier(cb, g_vk.sc_images[cur_frame.image], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    VkBufferImageCopy rg = {.bufferOffset = out->off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                            .imageExtent = {w, h, 1}};
    vkCmdCopyImageToBuffer(cb, g_vk.sc_images[cur_frame.image], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out->buf, 1, &rg);
    image_barrier(cb, g_vk.sc_images[cur_frame.image], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
}

// Copies a BGRA text block (hle_osd.cpp) into the window image at (x, y); skipped when it does not fit.
static void osd_blit(VkCommandBuffer cb, const uint32_t *px, int bw, int bh, int x, int y) {
    if (x < 0 || y < 0 || x + bw > (int)g_vk.sc_extent.width || y + bh > (int)g_vk.sc_extent.height) return;
    VkSpan s = vk_transient_copy(px, (uint64_t)bw * bh * 4, 16);
    if (g_vk.sc_format != VK_FORMAT_B8G8R8A8_UNORM && g_vk.sc_format != VK_FORMAT_B8G8R8A8_SRGB)
        for (uint32_t *q = (uint32_t *)s.ptr, *e = q + (size_t)bw * bh; q < e; q++)   // RGBA window image
            *q = (*q & 0xff00ff00u) | (*q >> 16 & 0xffu) | (*q & 0xffu) << 16;
    VkImage im = g_vk.sc_images[cur_frame.image];
    image_barrier(cb, im, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkBufferImageCopy rg = {.bufferOffset = s.off, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                            .imageOffset = {x, y, 0}, .imageExtent = {(uint32_t)bw, (uint32_t)bh, 1}};
    vkCmdCopyBufferToImage(cb, s.buf, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &rg);
    image_barrier(cb, im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
}
static char g_fps_text[96];   // backend_present's 0.5 s fps line (window title + fps readout)

// End rendering into the window image and present it. shot_path: also save the window image as PNG.
void vk_frame_end(const char *shot_path) {
    if (!cur_frame.ok) return;
    VkCommandBuffer cb = vk_cmd();
    vkCmdEndRendering(cb);
    {   // runtime option banner (hle_osd.cpp: "Skip opening: ON/OFF" while the title screen is up), bottom left
        extern const uint32_t *osd_banner(int target_h, int *w, int *h);
        int bw = 0, bh = 0;
        const uint32_t *px = osd_banner((int)g_vk.sc_extent.height, &bw, &bh);
        if (px) osd_blit(cb, px, bw, bh, bh / 2, (int)g_vk.sc_extent.height - bh - bh / 2);
    }
    {   // fps readout, top right (top left covers the hearts): while fullscreen (no title bar) or after F3 (shim_win.c)
        extern bool shim_fps_overlay(void);
        extern const uint32_t *osd_fps(const char *text, int target_h, int *w, int *h);
        int bw = 0, bh = 0;
        const uint32_t *px = shim_fps_overlay() ? osd_fps(g_fps_text, (int)g_vk.sc_extent.height, &bw, &bh) : NULL;
        if (px) osd_blit(cb, px, bw, bh, (int)g_vk.sc_extent.width - bw - bh / 3, bh / 3);
    }
    {   // Graphics menu (hle_osd.cpp: L3 + R3 / F10), centred
        extern const uint32_t *osd_menu(int target_h, int *w, int *h);
        int bw = 0, bh = 0;
        const uint32_t *px = osd_menu((int)g_vk.sc_extent.height, &bw, &bh);
        if (px) osd_blit(cb, px, bw, bh, ((int)g_vk.sc_extent.width - bw) / 2, ((int)g_vk.sc_extent.height - bh) / 2);
    }
    VkSpan shot = {0};
    if (shot_path) save_window_image(cb, &shot);
    image_barrier(cb, g_vk.sc_images[cur_frame.image], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    uint32_t img = cur_frame.image;
    if (!done_sem[img]) done_sem[img] = binary_semaphore();
    uint64_t seq = vk_pending_seq();
    vk_submit_extra(cur_frame.acquired, done_sem[img], VK_NULL_HANDLE);
    VkPresentInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1, .pWaitSemaphores = &done_sem[img],
                           .swapchainCount = 1, .pSwapchains = &g_vk.swapchain, .pImageIndices = &img};
    g_rt_op = "vkQueuePresentKHR", g_rt_op_ns = SDL_GetTicksNS();
    VkResult r = vkQueuePresentKHR(g_vk.queue, &pi);
    g_rt_op = NULL;
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) recreate_swapchain();
    cur_frame.ok = false;
    if (shot_path) {
        vk_wait_seq(seq);
        uint32_t w = g_vk.sc_extent.width, h = g_vk.sc_extent.height;
        uint8_t *px = shot.ptr;
        bool bgr = g_vk.sc_format == VK_FORMAT_B8G8R8A8_UNORM || g_vk.sc_format == VK_FORMAT_B8G8R8A8_SRGB;
        for (uint64_t i = 0; bgr && i < (uint64_t)w * h; i++) { uint8_t t = px[i * 4]; px[i * 4] = px[i * 4 + 2]; px[i * 4 + 2] = t; }
        for (uint64_t i = 0; i < (uint64_t)w * h; i++) px[i * 4 + 3] = 255;
        vk_save_png(shot_path, px, w, h);
    }
}

void vk_save_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h) {
    SDL_Surface *s = SDL_CreateSurfaceFrom((int)w, (int)h, SDL_PIXELFORMAT_ABGR8888, (void *)rgba, (int)w * 4);
    if (!s || !SDL_SavePNG(s, path)) LOG("[vk] saving %s failed: %s\n", path, SDL_GetError());
    else LOG("[vk] saved %s (%ux%u)\n", path, w, h);
    SDL_DestroySurface(s);
}

// ================================================================== game presentation (metal.m backend_present)
// Draws the game's image into the window with the widescreen crop, then the overlay menu.
extern void vk_present_draw(VkCommandBuffer cb, TexBackend *b, float vx, float vy, float vw, float vh, const float pv[8]);
void renderer_submit(void);
void renderer_set_scale(float sx, float sy);
uint64_t renderer_present_count(void);
void renderer_save_texture_tex(TexBackend *b, const char *path);

static uint64_t frames, total_frames, g_frame_counter;
uint64_t renderer_present_count(void) { return total_frames; }
uint64_t vk_frame_counter(void) { return g_frame_counter; }
static float g_shot_crop[4] = {0, 0, 1, 1};
static uint64_t g_held_window;            // frames not shown in this stats window (backend_present hold)
static uint64_t g_shown_with_skips;       // frames shown although small draws were skipped (not held, LARECOMP_HOLD_MIN)
static double g_last_shown_t, g_shown_gap_max;  // seconds: last frame actually presented, longest gap between them
static double last_fps_time;
static NvnTexture *g_last_presented;

const float *vk_shot_crop(void) { return g_shot_crop; }

void backend_present(NvnTexture *t);
void backend_represent(void) {
    if (g_last_presented) backend_present(g_last_presented);
    else SDL_Delay(16);
}

// LARECOMP_MEMREPORT_AT=<frame>[,<frame>...] (Steam Deck: 16 GB shared by CPU and GPU, 2026-10-16): where the memory
// goes. Vulkan heaps (VMA blocks per heap; imported host memory such as the guest heap is not in VMA) and the process's
// committed memory: on Windows per reservation (top 14 private ones, so the guest heap, the alias window, the driver and
// host malloc can be told apart by base address), on Linux /proc/self/status.
#ifdef _WIN32
#include <psapi.h>
#endif
uint32_t vma_shim_heaps(VmaAllocator a, uint64_t *bytes, uint32_t *count, uint64_t *size, uint32_t *flags);  // shim_vma.c
static void vk_mem_report(void) {
    uint64_t hb[VK_MAX_MEMORY_HEAPS], hs[VK_MAX_MEMORY_HEAPS];
    uint32_t hc[VK_MAX_MEMORY_HEAPS], hf[VK_MAX_MEMORY_HEAPS];
    uint32_t nh = vma_shim_heaps(g_vk.vma, hb, hc, hs, hf);
    for (uint32_t h = 0; h < nh; h++)
        LOG("[mem] vk heap %u%s: %llu MB in %u allocations (shim; imported host memory not included), heap %llu MB\n", h,
            (hf[h] & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? " device-local" : " host", (unsigned long long)(hb[h] >> 20), hc[h],
            (unsigned long long)(hs[h] >> 20));
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pc = {.cb = sizeof pc};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&pc, sizeof pc))
        LOG("[mem] process: private %llu MB (peak commit %llu MB), working set %llu MB (peak %llu MB)\n",
            (unsigned long long)(pc.PrivateUsage >> 20), (unsigned long long)(pc.PeakPagefileUsage >> 20),
            (unsigned long long)(pc.WorkingSetSize >> 20), (unsigned long long)(pc.PeakWorkingSetSize >> 20));
    enum { NTOP = 14 };
    struct { uintptr_t base; uint64_t commit, res; } top[NTOP] = {{0}}, cur = {0};
    uint64_t priv = 0, mapped = 0, image = 0, small = 0, nsmall = 0;
    MEMORY_BASIC_INFORMATION m;
    for (uintptr_t a = 0x10000;; a = (uintptr_t)m.BaseAddress + m.RegionSize) {
        int ok = VirtualQuery((void *)a, &m, sizeof m) == sizeof m && (uintptr_t)m.BaseAddress + m.RegionSize > a;
        uintptr_t ab = ok && m.State != MEM_FREE ? (uintptr_t)m.AllocationBase : 0;
        if (!ok || (ab && ab != cur.base)) {   // a reservation ended: rank it
            if (cur.commit) {
                if (cur.commit < (64ull << 20)) { small += cur.commit; nsmall++; }
                int i = NTOP;
                while (i > 0 && top[i - 1].commit < cur.commit) { if (i < NTOP) top[i] = top[i - 1]; i--; }
                if (i < NTOP) top[i] = cur;
            }
            cur.base = ab; cur.commit = cur.res = 0;
        }
        if (!ok) break;
        if (!ab) continue;
        cur.res += m.RegionSize;
        if (m.State != MEM_COMMIT) continue;
        if (m.Type == MEM_PRIVATE) { priv += m.RegionSize; cur.commit += m.RegionSize; }
        else if (m.Type == MEM_MAPPED) mapped += m.RegionSize;
        else image += m.RegionSize;
    }
    LOG("[mem] committed: private %llu MB (%llu MB in %llu reservations under 64 MB each), mapped %llu MB, image %llu MB\n",
        (unsigned long long)(priv >> 20), (unsigned long long)(small >> 20), (unsigned long long)nsmall,
        (unsigned long long)(mapped >> 20), (unsigned long long)(image >> 20));
    for (int i = 0; i < NTOP && top[i].commit; i++)
        LOG("[mem]   reservation %llx: %llu MB committed of %llu MB\n", (unsigned long long)top[i].base,
            (unsigned long long)(top[i].commit >> 20), (unsigned long long)(top[i].res >> 20));
    // heap blocks (malloc = the CRT's process heap; drivers may keep their own heaps or VirtualAlloc directly): busy bytes
    // per heap, and for blocks >= 1 MB the most common sizes (64 KB buckets), to tell what the many 4-5 MB reservations are
    HANDLE heaps[64];
    DWORD nheap = GetProcessHeaps(64, heaps);
    for (DWORD hi = 0; hi < nheap && hi < 64; hi++) {
        enum { NB = 4096 };
        static uint32_t bucket[NB];
        memset(bucket, 0, sizeof bucket);
        uint64_t busy = 0, nbusy = 0, big = 0, nbig = 0;
        PROCESS_HEAP_ENTRY e = {0};
        if (!HeapLock(heaps[hi])) continue;
        while (HeapWalk(heaps[hi], &e))
            if (e.wFlags & PROCESS_HEAP_ENTRY_BUSY) {
                busy += e.cbData; nbusy++;
                if (e.cbData >= (1u << 20)) { big += e.cbData; nbig++; uint64_t b = e.cbData >> 16; bucket[b < NB ? b : NB - 1]++; }
            }
        HeapUnlock(heaps[hi]);
        if (busy < (16ull << 20)) continue;
        LOG("[mem] heap %p%s: %llu MB busy in %llu blocks, %llu MB in %llu blocks >= 1 MB\n", heaps[hi],
            heaps[hi] == GetProcessHeap() ? " (process heap)" : "", (unsigned long long)(busy >> 20), (unsigned long long)nbusy,
            (unsigned long long)(big >> 20), (unsigned long long)nbig);
        for (int k = 0; k < 6; k++) {
            int bi = -1;
            for (int b = 16; b < NB; b++) if (bucket[b] && (bi < 0 || bucket[b] > bucket[bi])) bi = b;
            if (bi < 0) break;
            LOG("[mem]   %u blocks of %llu-%llu KB\n", bucket[bi], (unsigned long long)bi * 64, (unsigned long long)bi * 64 + 63);
            bucket[bi] = 0;
        }
    }
#else
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    while (f && fgets(line, sizeof line, f))
        if (!strncmp(line, "VmRSS", 5) || !strncmp(line, "VmHWM", 5) || !strncmp(line, "RssAnon", 7) || !strncmp(line, "RssFile", 7) ||
            !strncmp(line, "RssShmem", 8) || !strncmp(line, "VmSwap", 6) || !strncmp(line, "VmData", 6))
            LOG("[mem] process %s", line);
    if (f) fclose(f);
#endif
}

void backend_present(NvnTexture *t) {
    g_last_presented = t;
    TexBackend *b = vk_tex(t, true);
    b->presented = true;
    vk_end_pass();
    static const float black[4] = {0, 0, 0, 1};
    uint64_t t0 = SDL_GetTicksNS();
    // A frame that skipped draws or dispatches whose shaders / pipelines were still being built is not shown: the window
    // keeps the previous frame. Showing it made whole frames black when the skipped draw was a final pass (run199 black
    // frames 2361 / 5427: 36 / 43 skips); building inline instead stalled 1.9-2.7 s (run201 / run202). Workers build in
    // parallel, so a burst of new pipelines costs a few held frames. At most 30 frames in a row are held.
    // LARECOMP_NO_HOLD=1: show them anyway (A/B). "[hold]" lines + "held N, shown gap max" in the stats line.
    extern uint64_t g_skip_prog_frame, g_skip_pipe_frame, g_skip_comp_frame, g_draws_frame;
    static int hold_on = -1;
    if (hold_on < 0) hold_on = getenv("LARECOMP_NO_HOLD") == NULL;
    static bool shown_any;
    static uint64_t held_run, held_from, held_logs, held_sk[3];  // held_sk: skipped programs / pipelines / computes in the run
    // Only frames whose missing work matters are held (user 2026-10-15 "micro stutters": 1-2 skipped small draws held
    // the whole frame = a repeated frame). Held: any program / compute skip, any skipped fullscreen-ish draw (<= 6
    // vertices: post passes, tile lighting; a skipped final pass = the run199 black frames), or more than
    // LARECOMP_HOLD_MIN (default 2) skipped pipelines. LARECOMP_HOLD_MIN=0 = the old rule (hold on any skip).
    extern uint64_t g_skip_full_frame;
    static long hold_min = -1;
    if (hold_min < 0) { const char *e = getenv("LARECOMP_HOLD_MIN"); hold_min = e ? atol(e) : 2; }
    bool held = hold_on && shown_any && held_run < 30 &&
                (g_skip_prog_frame || g_skip_comp_frame || g_skip_full_frame || g_skip_pipe_frame > (uint64_t)hold_min);
    if (!held && g_skip_pipe_frame) g_shown_with_skips++;
    if (held) {
        if (!held_run++) { held_from = total_frames; held_sk[0] = held_sk[1] = held_sk[2] = 0; }
        held_sk[0] += g_skip_prog_frame; held_sk[1] += g_skip_pipe_frame; held_sk[2] += g_skip_comp_frame;
        g_held_window++;
    } else if (held_run) {
        if (held_logs++ < 60)
            LOG("[hold] frames %llu..%llu not shown (%llu frames; skipped: program %llu, pipeline %llu, compute %llu)\n",
                (unsigned long long)held_from, (unsigned long long)total_frames - 1, (unsigned long long)held_run,
                (unsigned long long)held_sk[0], (unsigned long long)held_sk[1], (unsigned long long)held_sk[2]);
        held_run = 0;
    }
    bool have = !held && vk_frame_begin(black);
    if (have) {
        shown_any = true;
        double ts = SDL_GetTicksNS() / 1e9;
        if (g_last_shown_t > 0 && ts - g_last_shown_t > g_shown_gap_max) g_shown_gap_max = ts - g_last_shown_t;
        g_last_shown_t = ts;
    }
    uint64_t nd = SDL_GetTicksNS() - t0;
    if (nd > 30000000) {
        static int warned;
        if (warned++ < 20) LOG("[vk] acquire blocked %llums\n", (unsigned long long)(nd / 1000000));
    }
    if (have) {
        double dw = g_vk.sc_extent.width, dh = g_vk.sc_extent.height;
        extern void widescreen_frame_end(uint64_t frame);
        widescreen_frame_end(total_frames);
        extern void widescreen_set_factor(float f);
        extern bool widescreen_allowed(void);
        // render scale + placement in the window: runtime/nvn/output_scale.c (Video tab "Internal resolution");
        // no GPU timer here (-1): dynamic resolution goes by the frame time
        OutScale os;
        output_scale(dw, dh, widescreen_allowed(), SDL_GetTicksNS() / 1e9, -1, &os);
        double sx = os.sx, sy = os.sy, vx = os.vx, vy = os.vy, vw = os.vw, vh = os.vh, shape = os.shape;
        // interiors / dungeons: their 16:9 centre, or the part inside a room wider than one screen
        extern void widescreen_present_crop(uint64_t, double, float *, double *, double *, double *, double *);
        float crop[4];  // uv origin and size
        widescreen_present_crop(total_frames, shape, crop, &vx, &vy, &vw, &vh);
        // the window's crop rect (nvnWindowSetCrop, guest pixels): TotK in handheld mode (TOTK_HANDHELD=1) renders 1280x720 into
        // the top-left of its 1920x1080 window textures and crops to it; without this the window showed that corner 1:1 (run393)
        extern int g_present_crop[4];
        int pc[4]; memcpy(pc, g_present_crop, sizeof pc);
        if (pc[2] > 0 && pc[3] > 0 && t->width && t->height &&
            (pc[0] || pc[1] || (uint32_t)pc[2] != t->width || (uint32_t)pc[3] != t->height)) {
            crop[0] = (float)pc[0] / t->width; crop[1] = (float)pc[1] / t->height;
            crop[2] = (float)pc[2] / t->width; crop[3] = (float)pc[3] / t->height;
        }
        widescreen_set_factor((float)shape);
        renderer_set_scale((float)sx, (float)sy);
        {   // TOTK_OUT_SCALE: the 1920x1080 output canvas at the window's resolution (textures.c kind 3)
            extern void renderer_set_out_scale(float s);
            double s = vh / 1080.0;
            renderer_set_out_scale((float)(s < 1.0 ? 1.0 : s > 4.0 ? 4.0 : s));
        }
        VkCommandBuffer cb = vk_cmd();
        float pv[8] = {renderer_origin_lower_left() ? 1.0f : 0.0f, 0, 0, 0, crop[0], crop[1], crop[2], crop[3]};
        memcpy(g_shot_crop, crop, sizeof crop);
        vk_present_draw(cb, b, (float)vx, (float)vy, (float)vw, (float)vh, pv);
        static long menu_at = -1;
        if (menu_at < 0) menu_at = getenv("LARECOMP_MENU_AT") ? atol(getenv("LARECOMP_MENU_AT")) : 0;
        if (menu_at && (long)total_frames + 1 == menu_at) { extern void menu_test_open(void); menu_test_open(); }
        menu_render((void *)cb, g_vk.sc_extent.width, g_vk.sc_extent.height);
        static long win_shot = -1;
        if (win_shot < 0) win_shot = getenv("LARECOMP_WINDOW_SHOT") ? atol(getenv("LARECOMP_WINDOW_SHOT")) : 0;
        bool shot = win_shot && (long)total_frames + 1 == win_shot;
        // TotK: TOTK_SHOT_EVERY=N also saves build/window_<frame>.png every N frames (watch a scene progress in one run)
        static long shot_every = -1;
        if (shot_every < 0) shot_every = getenv("TOTK_SHOT_EVERY") ? atol(getenv("TOTK_SHOT_EVERY")) : 0;
        static char shot_path[64];
        const char *shot_to = shot ? "build/window.png" : NULL;
        // TOTK_SHOT_RANGE=a-b: every frame from a to b too (flicker that changes each frame)
        static long range_a = -1, range_b;
        if (range_a < 0) { const char *e = getenv("TOTK_SHOT_RANGE"); range_a = e ? strtol(e, (char **)&e, 10) : 0; range_b = e && *e == '-' ? atol(e + 1) : 0; }
        if (!shot_to && ((shot_every > 0 && ((long)total_frames + 1) % shot_every == 0) ||
                         (range_b && (long)total_frames + 1 >= range_a && (long)total_frames + 1 <= range_b))) {
            snprintf(shot_path, sizeof shot_path, "build/window_%05ld.png", (long)total_frames + 1);
            shot_to = shot_path;
        }
        vk_frame_end(shot_to);
        if (shot && getenv("LARECOMP_EXIT_AFTER_SHOT")) { fflush(stdout); fflush(stderr); _exit(0); }
    } else {
        renderer_submit();
    }
    extern void vk_prof_frame(uint64_t frame, uint64_t present_ns);
    vk_prof_frame(total_frames, SDL_GetTicksNS() - t0);
    frames++;
    total_frames++;
    atomic_store(&g_last_frame_ns, SDL_GetTicksNS());
#ifndef _WIN32
    if (!g_present_tid) g_present_tid = (int)syscall(SYS_gettid);
#endif
    g_frame_counter++;
    extern void input_frame(uint64_t frame);
    input_frame(total_frames);
    static long slow_ms = -1;
    if (slow_ms < 0) slow_ms = getenv("LARECOMP_SLOW_FRAME_MS") ? atol(getenv("LARECOMP_SLOW_FRAME_MS")) : 0;
    if (slow_ms > 0 && total_frames > 2600) SDL_Delay((Uint32)slow_ms);
    extern void fnprof_frame(uint64_t frame);
    fnprof_frame(total_frames);
    extern void warp_frame(uint64_t frame);
    warp_frame(total_frames);
    extern void reboot_test_frame(uint64_t frame);  // runtime/main.c (LARECOMP_REBOOT_AT)
    reboot_test_frame(total_frames);
    extern void save_editor_frame(uint64_t frame);
    save_editor_frame(total_frames);
    extern void vk_level_change_check(void);
    vk_level_change_check();
    static long shot_every = -1;
    if (shot_every < 0) { const char *e = getenv("LARECOMP_SCREENSHOT_EVERY"); shot_every = e ? atol(e) : 0; }
    extern bool g_debug_frame;
    // LARECOMP_DEBUG_FRAME=<frame>[,<frame>...] (ascending, up to 8): debug-dump those frames
    static long debug_at = -1, debug_list[8];
    static int debug_cnt, debug_i;
    if (debug_at < 0) {
        const char *e = getenv("LARECOMP_DEBUG_FRAME");
        for (const char *p = e; p && *p && debug_cnt < 8;) {
            debug_list[debug_cnt++] = strtol(p, (char **)&p, 10);
            if (*p == ',') p++; else break;
        }
        debug_at = debug_cnt ? debug_list[0] : 0;
    }
    extern volatile int g_debug_frame_request;
    if (g_debug_frame_request) { g_debug_frame_request = 0; debug_at = (long)total_frames + 1; }
    // LARECOMP_DEBUG_FRAMES=<n> (with LARECOMP_DEBUG_FRAME / F9): dump n consecutive frames (build/pass_<frame>_NNN.png)
    static long debug_n = -1;
    if (debug_n < 0) { const char *e = getenv("LARECOMP_DEBUG_FRAMES"); debug_n = e && atol(e) > 1 ? atol(e) : 1; }
    while (debug_i + 1 < debug_cnt && debug_at && total_frames >= (uint64_t)debug_at + (uint64_t)debug_n) debug_at = debug_list[++debug_i];
    g_debug_frame = debug_at && total_frames >= (uint64_t)debug_at && total_frames < (uint64_t)debug_at + (uint64_t)debug_n;
    {   // LARECOMP_GPUPROF_AT=<frame>[,<frame>...]: GPU time per pass of those frames ("[gpuprof]" in the log)
        extern bool g_gpuprof_frame;
        extern void vk_gpuprof_report(void);
        if (g_gpuprof_frame) { g_gpuprof_frame = false; vk_gpuprof_report(); }
        extern volatile int g_gpuprof_request;   // (exec.c vk_prof_frame: LARECOMP_GPUPROF_SLOW)
        if (g_gpuprof_request) {
            g_gpuprof_request = 0;
            g_gpuprof_frame = true;
            vk_mem_report();
            mkdir("build", 0755);
            mkdir("build/frames", 0755);
            char path[256];
            snprintf(path, sizeof path, "build/frames/slow_%05llu.png", (unsigned long long)total_frames);
            if (b) renderer_save_texture_tex(b, path);
        }
        const char *e = getenv("LARECOMP_GPUPROF_AT");
        for (const char *p = e; p && *p;) {
            long at = strtol(p, (char **)&p, 10);
            if (at == (long)total_frames) { g_gpuprof_frame = true; break; }
            if (*p == ',') p++; else break;
        }
    }
    {   // LARECOMP_MEMREPORT_AT=<frame>[,<frame>...]: memory breakdown ("[mem]" in the log; vk_mem_report above)
        const char *e = getenv("LARECOMP_MEMREPORT_AT");
        for (const char *p = e; p && *p;) {
            long at = strtol(p, (char **)&p, 10);
            if (at == (long)total_frames) { vk_mem_report(); break; }
            if (*p == ',') p++; else break;
        }
    }
    if (g_debug_frame) LOG("===== debug frame %llu =====\n", (unsigned long long)total_frames);
    {   // LARECOMP_SHOT_AT=<frame>,<frame>,... (ascending): the game image at those frames -> build/frames/frame_N.png
        static const char *next;
        static int init;
        if (!init) { init = 1; next = getenv("LARECOMP_SHOT_AT"); if (next) { mkdir("build", 0755); mkdir("build/frames", 0755); } }
        while (next && *next) {
            long at = strtol(next, (char **)&next, 10);
            if ((long)total_frames < at) { while (next > getenv("LARECOMP_SHOT_AT") && next[-1] != ',') next--; break; }
            if ((long)total_frames == at) {
                char path[256];
                snprintf(path, sizeof path, "build/frames/frame_%05llu.png", (unsigned long long)total_frames);
                renderer_save_texture_tex(b, path);
            }
            if (*next == ',') next++;
        }
    }
    {   // LARECOMP_BLACKCHECK=1 (user 22:23: "the screen randomly goes completely black for like .2 of a second"): read a 4x4
        // grid of the presented game image every frame (one GPU sync per frame) and log frames that are all black
        static int on = -1;
        if (on < 0) on = getenv("LARECOMP_BLACKCHECK") != NULL;
        if (on && !held && b && b->image && b->width >= 8 && b->height >= 8 && !(b->aspect & ~VK_IMAGE_ASPECT_COLOR_BIT)) {
            VkSpan s = vk_readback(16 * 16);
            VkCommandBuffer cb = vk_cmd();
            vk_full_barrier(cb);
            VkBufferImageCopy rg[16];
            for (int i = 0; i < 16; i++)
                rg[i] = (VkBufferImageCopy){.bufferOffset = s.off + (VkDeviceSize)i * 16, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                            .imageOffset = {(int32_t)(b->width * (1 + 2 * (i & 3)) / 8), (int32_t)(b->height * (1 + 2 * (i >> 2)) / 8), 0},
                                            .imageExtent = {1, 1, 1}};
            vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, s.buf, 16, rg);
            renderer_flush();
            int mx = 0;
            for (int i = 0; i < 16; i++) for (int k = 0; k < 3; k++) { int v = ((const uint8_t *)s.ptr)[i * 16 + k]; if (v > mx) mx = v; }
            static uint64_t run_start, nblack;
            if (mx <= 2) {
                if (!run_start) {
                    run_start = total_frames;
                    // LARECOMP_BLACKCHECK=capture: the first 2 black runs after boot (frame > 1000) also dump the next frame
                    // (as F9 does): a flash lasts ~4 frames, so that frame is still black and its passes show where it goes black
                    static int captures;
                    const char *e = getenv("LARECOMP_BLACKCHECK");
                    if (e && !strcmp(e, "capture") && total_frames > 1000 && captures < 2) {
                        extern volatile int g_debug_frame_request;
                        g_debug_frame_request = 1;
                        captures++;
                        LOG("[black] frame %llu black: capturing the next frame\n", (unsigned long long)total_frames);
                    }
                }
                nblack++;
                extern uint64_t g_skip_prog_frame, g_skip_pipe_frame, g_draws_frame;  // (exec.c, this frame)
                if (total_frames > 1000)
                    LOG("[black] frame %llu: %llu draws, skipped (program not built %llu, pipeline not built %llu)\n",
                        (unsigned long long)total_frames, (unsigned long long)g_draws_frame, (unsigned long long)g_skip_prog_frame,
                        (unsigned long long)g_skip_pipe_frame);
            } else if (run_start) {
                LOG("[black] frames %llu..%llu all black (%llu frames; %llu black frames so far)\n", (unsigned long long)run_start,
                    (unsigned long long)total_frames - 1, (unsigned long long)(total_frames - run_start), (unsigned long long)nblack);
                run_start = 0;
            }
        }
    }
    {   // LARECOMP_FLICKCHECK=1 (user 2026-10-19: UI / mesh flicker seen live, hidden by captures that sync CPU and GPU): a
        // 48x27 grid of the presented game image goes into a host ring every frame WITHOUT waiting; once its submission is done
        // (a few frames later) a frame whose samples jump away from both neighbours while those two agree is logged ("[flick]").
        enum { FX = 64, FY = 36, FN = FX * FY, FK = 8, FT = 48 };
        static int on = -1;
        static VkBuffer fbuf;
        static VmaAllocation fmem;
        static uint8_t *fptr;
        static uint64_t fseq[FK], fframe[FK], fhead, ftail, hframe[3], nflick, nchecked, win_from;
        static uint8_t hist[3][FN * 3];
        static int nh;
        static FILE *fdump;  // LARECOMP_FLICKCHECK=dump: every checked frame's samples -> build/flick.bin (u64 frame + FN x RGB)
        // LARECOMP_FLICKCROP=<cx>,<cy> (grid cell): a CW x CH crop around that cell rides along each frame; when the cell flips,
        // the crops of the frames before / at / after go to build/flick_crops.bin (u64 frame, u32 w, u32 h, RGBA x3), max 24
        enum { CW = 320, CH = 160 };
        static int ccx = -1, ccy = -1, ncrops;
        static VkBuffer cbuf;
        static VmaAllocation cmem;
        static uint8_t *cptr, chist[3][CW * CH * 4];
        static FILE *fcrop;
        if (on < 0) {
            on = getenv("LARECOMP_FLICKCHECK") != NULL;
            if (on && !strcmp(getenv("LARECOMP_FLICKCHECK"), "dump")) { mkdir("build", 0755); fdump = fopen("build/flick.bin", "wb"); }
            const char *ce = getenv("LARECOMP_FLICKCROP");
            if (on && ce && sscanf(ce, "%d,%d", &ccx, &ccy) == 2 && ccx >= 0 && ccx < FX && ccy >= 0 && ccy < FY) {
                VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)FK * CW * CH * 4,
                                         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
                VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                              .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
                VmaAllocationInfo info;
                if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &cbuf, &cmem, &info) == VK_SUCCESS) cptr = info.pMappedData;
                mkdir("build", 0755);
                fcrop = fopen("build/flick_crops.bin", "wb");
            } else ccx = -1;
            if (on) {
                VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)FK * FN * 16,
                                         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
                VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                              .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
                VmaAllocationInfo info;
                if (vmaCreateBuffer(g_vk.vma, &bi, &ai, &fbuf, &fmem, &info) == VK_SUCCESS) fptr = info.pMappedData;
                LOG("[flick] check on (%dx%d samples per frame, threshold %d)\n", FX, FY, FT);
            }
        }
        if (on && fptr) {
            extern uint64_t vk_completed_seq(void);
            extern uint64_t renderer_fence_seq(void);
            while (ftail < fhead && vk_completed_seq() >= fseq[ftail % FK]) {
                const uint8_t *p = fptr + (ftail % FK) * (uint64_t)FN * 16;
                memmove(hist[0], hist[1], sizeof hist[0] * 2);
                hframe[0] = hframe[1], hframe[1] = hframe[2], hframe[2] = fframe[ftail % FK];
                for (int i = 0; i < FN; i++) memcpy(&hist[2][i * 3], p + i * 16, 3);
                if (fdump) { fwrite(&hframe[2], 8, 1, fdump); fwrite(hist[2], FN * 3, 1, fdump); }
                if (cptr) {
                    memmove(chist[0], chist[1], sizeof chist[0] * 2);
                    memcpy(chist[2], cptr + (ftail % FK) * (uint64_t)CW * CH * 4, sizeof chist[2]);
                }
                ftail++;
                if (++nh < 3 || hframe[1] != hframe[0] + 1 || hframe[2] != hframe[1] + 1) continue;
                int n = 0, x0 = FX, y0 = FY, x1 = -1, y1 = -1;
                for (int i = 0; i < FN; i++) {
                    int dab = 0, dbc = 0, dac = 0;
                    for (int k = 0; k < 3; k++) {
                        int a = hist[0][i * 3 + k], bb = hist[1][i * 3 + k], cc = hist[2][i * 3 + k];
                        dab += abs(bb - a), dbc += abs(bb - cc), dac += abs(a - cc);
                    }
                    if (dab > FT && dbc > FT && dac <= FT / 2) {
                        n++;
                        int x = i % FX, y = i / FX;
                        if (x < x0) x0 = x;
                        if (x > x1) x1 = x;
                        if (y < y0) y0 = y;
                        if (y > y1) y1 = y;
                    }
                }
                nchecked++;
                if (cptr && fcrop && ncrops < 24) {
                    int i = ccy * FX + ccx, dab = 0, dbc = 0, dac = 0;
                    for (int k = 0; k < 3; k++) {
                        int a = hist[0][i * 3 + k], bb = hist[1][i * 3 + k], cc = hist[2][i * 3 + k];
                        dab += abs(bb - a), dbc += abs(bb - cc), dac += abs(a - cc);
                    }
                    if (dab > FT && dbc > FT && dac <= FT / 2) {
                        uint32_t wh[2] = {CW, CH};
                        fwrite(&hframe[1], 8, 1, fcrop);
                        fwrite(wh, 4, 2, fcrop);
                        fwrite(chist, sizeof chist, 1, fcrop);
                        fflush(fcrop);
                        ncrops++;
                    }
                }
                if (n >= 3) {
                    static int nlog;
                    nflick++;
                    if (nlog++ < 400)
                        LOG("[flick] frame %llu: %d/%d samples transient, grid box (%d,%d)-(%d,%d)\n", (unsigned long long)hframe[1], n, FN,
                            x0, y0, x1, y1);
                }
                if (hframe[1] - win_from >= 600) {
                    LOG("[flick] frames %llu..%llu: %llu flicker frames of %llu checked\n", (unsigned long long)win_from,
                        (unsigned long long)hframe[1], (unsigned long long)nflick, (unsigned long long)nchecked);
                    win_from = hframe[1] + 1, nflick = nchecked = 0;
                }
            }
            bool fmt4 = b && (b->format == VK_FORMAT_R8G8B8A8_UNORM || b->format == VK_FORMAT_R8G8B8A8_SRGB ||
                              b->format == VK_FORMAT_B8G8R8A8_UNORM || b->format == VK_FORMAT_B8G8R8A8_SRGB ||
                              b->format == VK_FORMAT_A2B10G10R10_UNORM_PACK32);
            if (b && !fmt4) { static int told; if (!told++) LOG("[flick] presented format %d not checked\n", (int)b->format); }
            if (!held && fmt4 && b->image && b->width >= FX && b->height >= FY && fhead - ftail < FK) {
                VkCommandBuffer cb = vk_cmd();
                vk_full_barrier(cb);
                static VkBufferImageCopy rg[FN];
                uint64_t base = (fhead % FK) * (uint64_t)FN * 16;
                for (int i = 0; i < FN; i++)
                    rg[i] = (VkBufferImageCopy){.bufferOffset = base + (VkDeviceSize)i * 16, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                                .imageOffset = {(int32_t)(b->width * (1 + 2 * (i % FX)) / (2 * FX)),
                                                                (int32_t)(b->height * (1 + 2 * (i / FX)) / (2 * FY)), 0},
                                                .imageExtent = {1, 1, 1}};
                vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, fbuf, FN, rg);
                if (cptr && b->width >= CW && b->height >= CH) {
                    int32_t x = (int32_t)(b->width * (1 + 2 * ccx) / (2 * FX)) - CW / 2, y = (int32_t)(b->height * (1 + 2 * ccy) / (2 * FY)) - CH / 2;
                    if (x < 0) x = 0;
                    if (y < 0) y = 0;
                    if (x + CW > (int32_t)b->width) x = (int32_t)b->width - CW;
                    if (y + CH > (int32_t)b->height) y = (int32_t)b->height - CH;
                    VkBufferImageCopy cr = {.bufferOffset = (fhead % FK) * (VkDeviceSize)CW * CH * 4, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                            .imageOffset = {x, y, 0}, .imageExtent = {CW, CH, 1}};
                    vkCmdCopyImageToBuffer(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, cbuf, 1, &cr);
                }
                fseq[fhead % FK] = renderer_fence_seq();
                fframe[fhead % FK] = total_frames;
                fhead++;
            }
        }
    }
    {   // LARECOMP_FLICKVID=<from>-<to> (user 2026-10-19: "3d mesh flickering", too small for the 64x36 grid): every presented
        // frame in that range, filtered down to 320x180 RGBA8 without waiting, -> build/flickvid.bin (u64 frame + VW*VH*4)
        // once its submission is done. Frames are dropped (gap in the numbers) if the GPU falls VQ frames behind.
        enum { VW = 320, VH = 180, VQ = 16 };
        static long va = -1, vb;
        static VkImage vimg;
        static VmaAllocation vimem, vbmem;
        static VkBuffer vbuf;
        static uint8_t *vptr;
        static uint64_t vseq[VQ], vfr[VQ], vhead, vtail;
        static FILE *vf;
        extern uint64_t vk_completed_seq(void);
        extern uint64_t renderer_fence_seq(void);
        if (va < 0) {
            const char *e = getenv("LARECOMP_FLICKVID");
            if (!e || sscanf(e, "%ld-%ld", &va, &vb) != 2 || vb <= va) va = vb = 0;
            if (vb) {
                VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                                        .extent = {VW, VH, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                                        .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
                VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
                VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)VQ * VW * VH * 4,
                                         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
                VmaAllocationCreateInfo bai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                               .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
                VmaAllocationInfo info;
                mkdir("build", 0755);
                if (vmaCreateImage(g_vk.vma, &ci, &ai, &vimg, &vimem, NULL) != VK_SUCCESS ||
                    vmaCreateBuffer(g_vk.vma, &bi, &bai, &vbuf, &vbmem, &info) != VK_SUCCESS || !(vf = fopen("build/flickvid.bin", "wb")))
                    va = vb = 0;
                else {
                    vptr = info.pMappedData;
                    vk_image_init_layout(vk_cmd(), vimg, VK_IMAGE_ASPECT_COLOR_BIT);
                    LOG("[flickvid] frames %ld..%ld at %dx%d -> build/flickvid.bin\n", va, vb, VW, VH);
                }
            }
        }
        if (vb && vf) {
            while (vtail < vhead && vk_completed_seq() >= vseq[vtail % VQ]) {
                fwrite(&vfr[vtail % VQ], 8, 1, vf);
                fwrite(vptr + (vtail % VQ) * (uint64_t)VW * VH * 4, (size_t)VW * VH * 4, 1, vf);
                vtail++;
            }
            bool ok = b && b->image && (b->format == VK_FORMAT_R8G8B8A8_UNORM || b->format == VK_FORMAT_R8G8B8A8_SRGB ||
                                        b->format == VK_FORMAT_B8G8R8A8_UNORM || b->format == VK_FORMAT_B8G8R8A8_SRGB ||
                                        b->format == VK_FORMAT_A2B10G10R10_UNORM_PACK32);
            if (!held && ok && (long)total_frames >= va && (long)total_frames < vb && vhead - vtail < VQ) {
                VkCommandBuffer cb = vk_cmd();
                vk_full_barrier(cb);
                VkImageBlit bl = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)b->width, (int32_t)b->height, 1}},
                                  .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {VW, VH, 1}}};
                vkCmdBlitImage(cb, b->image, VK_IMAGE_LAYOUT_GENERAL, vimg, VK_IMAGE_LAYOUT_GENERAL, 1, &bl, VK_FILTER_LINEAR);
                vk_full_barrier(cb);
                VkBufferImageCopy c = {.bufferOffset = (vhead % VQ) * (VkDeviceSize)VW * VH * 4, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                       .imageExtent = {VW, VH, 1}};
                vkCmdCopyImageToBuffer(cb, vimg, VK_IMAGE_LAYOUT_GENERAL, vbuf, 1, &c);
                vseq[vhead % VQ] = renderer_fence_seq();
                vfr[vhead % VQ] = total_frames;
                vhead++;
            }
            if ((long)total_frames >= vb && vtail == vhead) {
                fclose(vf);
                vf = NULL;
                LOG("[flickvid] done, %llu frames written\n", (unsigned long long)vtail);
            }
        }
    }
    {   // per-frame counters for the black check above
        g_skip_prog_frame = g_skip_pipe_frame = g_skip_comp_frame = g_draws_frame = g_skip_full_frame = 0;
    }
    static long shot_from = -1;
    if (shot_from < 0) { const char *e = getenv("LARECOMP_SCREENSHOT_FROM"); shot_from = e ? atol(e) : 0; }
    if (shot_every && total_frames % shot_every == 0 && (long)total_frames >= shot_from) {
        char path[256];
        if (shot_from) mkdir("build/frames", 0755);
        snprintf(path, sizeof path, shot_from ? "build/frames/frame_%05llu.png" : "build/frame_%05llu.png",
                 (unsigned long long)total_frames);
        renderer_save_texture_tex(b, path);
        // a series (SCREENSHOT_FROM) runs on until LARECOMP_WINDOW_SHOT exits
        if (getenv("LARECOMP_EXIT_AFTER_SHOT") && !shot_from) { fflush(stderr); _exit(0); }
    }
    {   // window title: shown fps (held frames excluded) and the worst gap between shown frames, every 0.5 s
        // (user 2026-10-15: "add an fps counter to the window top bar so i can monitor the fps")
        static const char *cap;
        static char cap_buf[32];
        if (!cap) {
            const char *f = getenv("TOTK_FPS");
            if (getenv("TOTK_UNLOCK_FPS")) cap = "uncapped";
            else if (f && atoi(f) > 0) { snprintf(cap_buf, sizeof cap_buf, "cap %d", atoi(f)); cap = cap_buf; }
            else cap = "cap 30 (Switch)";
        }
        static double t_from, last_t, worst;
        static uint64_t shown_n;
        double tn = SDL_GetTicksNS() / 1e9;
        if (have) {
            shown_n++;
            if (last_t > 0 && tn - last_t > worst) worst = tn - last_t;
            last_t = tn;
        }
        if (t_from == 0) t_from = tn;
        else if (tn - t_from >= 0.5) {
            char title[200];
            if (last_t > 0 && tn - last_t > worst) worst = tn - last_t;  // nothing shown lately (frames held): the stall so far
            int pdone, ptotal;
            extern int vk_precompile_progress(int *done, int *total);
            char pre[48] = "";
            if (vk_precompile_progress(&pdone, &ptotal) && ptotal)
                snprintf(pre, sizeof pre, "  |  precompiling shaders %d %%", pdone * 100 / ptotal);
            snprintf(title, sizeof title, "TotK recomp  |  %.1f fps  |  worst frame %.0f ms  |  %s%s", shown_n / (tn - t_from),
                     worst * 1000.0, cap, pre);
            SDL_SetWindowTitle(window, title);
            snprintf(g_fps_text, sizeof g_fps_text, "%.0f fps  |  worst %.0f ms", shown_n / (tn - t_from), worst * 1000.0);
            t_from = tn, shown_n = 0, worst = 0;
        }
    }
    double now = SDL_GetTicks() / 1000.0;
    static double fps_iv = -1, prev_present, max_gap;
    if (fps_iv < 0) fps_iv = getenv("LARECOMP_FPS_INTERVAL") ? atof(getenv("LARECOMP_FPS_INTERVAL")) : 2.0;
    if (prev_present > 0 && now - prev_present > max_gap) max_gap = now - prev_present;
    prev_present = now;
    if (now - last_fps_time >= fps_iv) {
        extern const char *game_level_name(void);
        if (last_fps_time > 0)
            LOG("[vk] t=%.1f frame %llu %.1f fps max %.0fms held %llu (not held %llu) shown %.1f fps gap %.0fms level %s (%s)\n", now,
                (unsigned long long)total_frames, frames / (now - last_fps_time), max_gap * 1000.0, (unsigned long long)g_held_window,
                (unsigned long long)g_shown_with_skips, (frames - g_held_window) / (now - last_fps_time), g_shown_gap_max * 1000.0, game_level_name(), renderer_stats());
#ifndef _WIN32
        if (last_fps_time > 0) {
            char sysb[640]; rt_sys_line(sysb, sizeof sysb); LOG("[sys] %s\n", sysb);
            extern uint64_t vma_shim_tag_bytes(int tag, uint32_t *count);
            uint32_t c1, c2, c3, c0;
            uint64_t b1 = vma_shim_tag_bytes(1, &c1), b2 = vma_shim_tag_bytes(2, &c2), b3 = vma_shim_tag_bytes(3, &c3), b0 = vma_shim_tag_bytes(0, &c0);
            LOG("[mem] live: astc->rgba8 %llu MB (%u images), other textures %llu MB (%u), block shadows %llu MB (%u), rest %llu MB (%u)\n",
                (unsigned long long)(b1 >> 20), c1, (unsigned long long)(b2 >> 20), c2, (unsigned long long)(b3 >> 20), c3,
                (unsigned long long)(b0 >> 20), c0);
            rt_cpu_line(sysb, sizeof sysb); LOG("[cpu] %s\n", sysb);
        }
#endif
        if (last_fps_time > 0) {  // "[vram]" (run571: 3-14 fps after slow image creates at a DRS change = VRAM full?): the OS
            // budget / usage of each device-local heap (VK_EXT_memory_budget, filled by NVIDIA without enabling it) + ours
            VkPhysicalDeviceMemoryBudgetPropertiesEXT bud = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
            VkPhysicalDeviceMemoryProperties2 mp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &bud};
            vkGetPhysicalDeviceMemoryProperties2(g_vk.phys, &mp);
            uint64_t hb[VK_MAX_MEMORY_HEAPS], hs[VK_MAX_MEMORY_HEAPS];
            uint32_t hc[VK_MAX_MEMORY_HEAPS], hf[VK_MAX_MEMORY_HEAPS];
            uint32_t nh = vma_shim_heaps(g_vk.vma, hb, hc, hs, hf);
            for (uint32_t h = 0; h < mp.memoryProperties.memoryHeapCount && h < nh; h++)
                if (mp.memoryProperties.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    LOG("[vram] heap %u: usage %llu MB of budget %llu MB (heap %llu MB); ours %llu MB in %u allocations\n", h,
                        (unsigned long long)(bud.heapUsage[h] >> 20), (unsigned long long)(bud.heapBudget[h] >> 20),
                        (unsigned long long)(mp.memoryProperties.memoryHeaps[h].size >> 20), (unsigned long long)(hb[h] >> 20), hc[h]);
            extern uint64_t vma_shim_tag_bytes(int tag, uint32_t *count);
            extern uint64_t vk_recycle_bytes(int *n);
            extern uint64_t vk_mirror_bytes(uint64_t *alias, int used[4]);
            uint32_t c1, c2, c3;
            int nrec, mu[4];
            uint64_t al, hm = vk_mirror_bytes(&al, mu), rec = vk_recycle_bytes(&nrec);
            LOG("[vram] mirrors used: heap %d MB in %d blocks of 64 MB, alias %d MB in %d blocks\n", mu[0], mu[2], mu[1], mu[3]);
            uint64_t b1 = vma_shim_tag_bytes(1, &c1), b2 = vma_shim_tag_bytes(2, &c2), b3 = vma_shim_tag_bytes(3, &c3);
            LOG("[vram] ours: astc->rgba8 %llu MB (%u), other images %llu MB (%u) of which recycle pool %llu MB (%d), block shadows "
                "%llu MB (%u), heap mirror %llu MB, alias mirror %llu MB\n", (unsigned long long)(b1 >> 20), c1, (unsigned long long)(b2 >> 20), c2,
                (unsigned long long)(rec >> 20), nrec, (unsigned long long)(b3 >> 20), c3, (unsigned long long)(hm >> 20), (unsigned long long)(al >> 20));
        }
        max_gap = 0;
        g_held_window = 0;
        g_shown_with_skips = 0;
        g_shown_gap_max = 0;
        frames = 0;
        last_fps_time = now;
    }
}

// ================================================================== UI thread
void vk_exec_init(void);
void vk_shaders_init(void);

// exit() from any thread: let the GPU finish, keep the pipeline cache, then end the process without the DLL
// teardown (driver / validation-layer threads can stall it for minutes while the game's threads still run).
static _Atomic uint64_t g_exit_req_ns;   // quick_exit_handler started (hang_watch ends a stuck exit)
#ifndef _WIN32
// Reports a stall once (no frame for LARECOMP_HANG_REPORT_S = 6 s): the last blocking Vulkan call, memory, recent submissions,
// every thread's kernel wait and the presenting thread's backtrace. Closes the game after LARECOMP_HANG_EXIT_S = 30 s without
// a frame (0 = never), and ends an exit that is stuck waiting for the GPU after 10 s.
static void *hang_watch(void *arg) {
    (void)arg;
#ifndef _WIN32
    { extern int pthread_setname_np(pthread_t, const char *); pthread_setname_np(pthread_self(), "vk-hang-watch"); }   // ([cpu] line)
#endif
    const char *e = getenv("LARECOMP_HANG_REPORT_S");
    double report_s = e ? atof(e) : 6.0;
    e = getenv("LARECOMP_HANG_EXIT_S");
    double exit_s = e ? atof(e) : 30.0;
    bool reported = false;
    for (;;) {
        SDL_Delay(500);
        uint64_t now = SDL_GetTicksNS(), x = atomic_load(&g_exit_req_ns);
        if (x && now - x > 10000000000ull) {
            LOG("[hang] the game has been closing for 10 s (waiting for the GPU): ending it now\n");
            fflush(stdout);
            fflush(stderr);
            _exit(76);
        }
        uint64_t last = atomic_load(&g_last_frame_ns);
        if (!last || x) continue;
        double idle = (now - last) / 1e9;
        if (idle < report_s) {
            if (reported) LOG("[hang] frames again\n");
            reported = false;
            continue;
        }
        if (!reported) {
            reported = true;
            const char *op = g_rt_op;
            LOG("[hang] no frame for %.1f s (after frame %llu); blocking Vulkan call in progress: %s", idle, (unsigned long long)total_frames,
                op ? op : "none");
            if (op) LOG(" (for %.1f s)", (now - g_rt_op_ns) / 1e9);
            char sysb[640];
            rt_sys_line(sysb, sizeof sysb);
            LOG("\n[hang] %s\n", sysb);
            extern void vk_devlost_dump(void);
            vk_devlost_dump();
            rt_threads_dump(g_present_tid);
            fflush(stdout);
        }
        if (exit_s > 0 && idle >= exit_s) {
            LOG("[hang] no frame for %.0f s: closing the game (LARECOMP_HANG_EXIT_S=0 keeps it running)\n", exit_s);
            char sysb[640];
            rt_sys_line(sysb, sizeof sysb);
            LOG("[hang] %s\n", sysb);
            rt_threads_dump(0);
            fflush(stdout);
            fflush(stderr);
            _exit(75);
        }
    }
    return NULL;
}
#endif

static void quick_exit_handler(void) {
    atomic_store(&g_exit_req_ns, SDL_GetTicksNS());
    static int trace = -1;
    if (trace < 0) trace = getenv("LARECOMP_VK_TRACE") != NULL;
    if (trace) LOG("[vk] exit: waiting for the GPU\n");
    if (g_vk.dev) {
        vkDeviceWaitIdle(g_vk.dev);
        if (trace) LOG("[vk] exit: saving the pipeline cache\n");
        vk_pipeline_cache_save();
    }
    if (trace) LOG("[vk] exit: terminating\n");
    fflush(stdout);
    fflush(stderr);
#if defined(_WIN32) || defined(__linux__)
    plat_terminate();
#endif
}

void ui_window_init(void) {
    if (window) return;
    atexit(quick_exit_handler);
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    settings_load();
    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) FATAL("SDL_Init: %s\n", SDL_GetError());
    int win_w = 1280, win_h = 720;
    if (getenv("LARECOMP_WINDOW")) sscanf(getenv("LARECOMP_WINDOW"), "%dx%d", &win_w, &win_h);
    SDL_WindowFlags wflags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;
    if (getenv("LARECOMP_HIDDEN")) wflags |= SDL_WINDOW_HIDDEN;
    window = SDL_CreateWindow("TotK recomp", win_w, win_h, wflags);
    if (!window) FATAL("SDL_CreateWindow: %s\n", SDL_GetError());
    rt_window_icon(window);
    // the Fullscreen setting applies from the start screen on (not only once the game runs); test windows stay windows
    if (g_settings.fullscreen && !getenv("LARECOMP_WINDOW") && !getenv("LARECOMP_HIDDEN")) SDL_SetWindowFullscreen(window, true);
    create_instance();
    if (!SDL_Vulkan_CreateSurface(window, g_vk.inst, NULL, &g_vk.surface)) FATAL("SDL_Vulkan_CreateSurface: %s\n", SDL_GetError());
    pick_device();
    create_device();
    create_swapchain();
    vk_exec_init();
    vk_shaders_init();
}

SDL_Window *ui_window(void) { return window; }

void backend_init(void) {
    while (!atomic_load(&ui_ready)) SDL_Delay(1);
    LOG("[vk] %s, swapchain %ux%u\n", g_vk.props.deviceName, g_vk.sc_extent.width, g_vk.sc_extent.height);
}

// ---- in-place reboot (runtime/main.c teardown; main thread, guest threads stopped)
void backend_reset(void) {
    if (!g_vk.dev) return;
    vkDeviceWaitIdle(g_vk.dev);
    extern void vk_textures_reset(void);
    extern void vk_exec_reset(void);
    vk_textures_reset();
    vk_exec_reset();
    g_last_presented = NULL;
}
void backend_guest_memory_reset(void) {}  // (exec.c re-imports the heap on first use)
void backend_clear_window(void) {
    static const float black[4] = {0, 0, 0, 1};
    if (g_vk.dev && vk_frame_begin(black)) vk_frame_end(NULL);
}

int ui_run(void) {
    ui_window_init();
    atomic_store(&ui_ready, true);
    for (;;) {
        if (rt_reboot_requested()) return rt_reboot_requested();  // (runtime/main.c tears the game down)
        SDL_Event e;
        while (SDL_WaitEventTimeout(&e, 16)) {
            if (e.type == SDL_EVENT_QUIT) {
                LOG("window closed\n");
                // Not exit(): it ran static destructors under the still-running guest threads, so every close (Steam's
                // SIGTERM, window X) crashed in HLE code (Deck runs A-E, R1-R3 of 2026-10-20: SIGSEGV / SIGABRT after
                // "window closed") before the pipeline cache was saved. This waits for the GPU, saves, then _exit.
                quick_exit_handler();
            }
            if (menu_handle_event(&e)) continue;
            input_handle_event(&e);
        }
        menu_apply_window_requests(window);
    }
}
