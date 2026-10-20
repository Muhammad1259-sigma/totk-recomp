// HLE: nn::oe (applet / operation environment). We present a docked console in boost performance mode, in focus,
// language en-US, with no pending notification messages. Setters stay default stubs (logged, return 0).
#include "hle.h"
#include <thread>

HLE_AS(oe_GetDesiredLanguage, "_ZN2nn2oe18GetDesiredLanguageEv") {   // nn::settings::LanguageCode = char[8] by value
    u64 v = 0; memcpy(&v, "en-US", 5); c->x[0] = v;
}
// TOTK_HANDHELD=1: report handheld mode (normal performance) instead: the game then renders at its handheld resolution (1280x720
// at most instead of 1600x900) with its handheld settings, the mode for a 1280x800 screen such as the Steam Deck's.
static bool oe_handheld() {
    static const bool h = [] { const char* e = getenv("TOTK_HANDHELD"); bool v = e && *e && strcmp(e, "0") != 0;
                               if (v) fprintf(stderr, "[oe] handheld mode (TOTK_HANDHELD)\n"); return v; }();
    return h;
}
HLE_AS(oe_GetOperationMode, "_ZN2nn2oe16GetOperationModeEv") { c->x[0] = oe_handheld() ? 0 : 1; }      // 0 handheld, 1 console
HLE_AS(oe_GetPerformanceMode, "_ZN2nn2oe18GetPerformanceModeEv") { c->x[0] = oe_handheld() ? 0 : 1; }  // 0 normal, 1 boost
HLE_AS(oe_GetCurrentFocusState, "_ZN2nn2oe20GetCurrentFocusStateEv") { c->x[0] = 1; }  // 1 in focus
HLE_AS(oe_GetExpectedVolumeBalance, "_ZN2nn2oe24GetExpectedVolumeBalanceEPfS1_") { *(float*)c->x[0] = 1.0f; *(float*)c->x[1] = 1.0f; }
HLE_AS(oe_GetProgramTotalActiveTime, "_ZN2nn2oe25GetProgramTotalActiveTimeEv") { c->x[0] = 0; }
HLE_AS(oe_TryPopNotificationMessage, "_ZN2nn2oe25TryPopNotificationMessageEPj") { c->x[0] = 0; }
HLE_AS(oe_PopNotificationMessage, "_ZN2nn2oe22PopNotificationMessageEv") {   // blocks on the real system: never returns a message
    for (;;) std::this_thread::sleep_for(std::chrono::hours(1));
}
