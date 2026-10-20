// HLE: nn::time. TotK reads the clock for autosaves and day/time bookkeeping; run123 crashed after the unimplemented
// StandardUserSystemClock::GetCurrentTime left its PosixTime unwritten (entering "Beneath Hyrule Castle" = autosave).
// Host clock = the console clock (UTC seconds), host time zone = the console's. Layouts follow Eden's
// core/hle/service/time (PosixTime s64; CalendarTime 8 bytes; CalendarAdditionalInfo 0x18; SteadyClockTimePoint
// 0x18; SystemClockContext 0x20; ClockSnapshot 0xD0).
#include "hle.h"
#include <windows.h>
#include <ctime>
#include <cstring>
#include <cstdio>

struct CalendarTime { s16 year; s8 month, day, hour, minute, second; u8 pad; };
static_assert(sizeof(CalendarTime) == 8);
struct CalendarAdditionalInfo { u32 day_of_week, day_of_year; char tz_name[8]; u32 is_dst; s32 utc_offset; };
static_assert(sizeof(CalendarAdditionalInfo) == 0x18);
struct SteadyClockTimePoint { s64 time_point; u8 source_id[16]; };
static_assert(sizeof(SteadyClockTimePoint) == 0x18);
struct SystemClockContext { s64 offset; SteadyClockTimePoint steady; };
struct ClockSnapshot {
    SystemClockContext user_context, network_context;   // 0x00, 0x20
    s64 user_time, network_time;                         // 0x40, 0x48
    CalendarTime user_cal, network_cal;                  // 0x50, 0x58
    CalendarAdditionalInfo user_info, network_info;      // 0x60, 0x78
    SteadyClockTimePoint steady;                         // 0x90
    char location_name[0x24];                            // 0xA8
    u8 automatic_correction, type, pad[2];               // 0xCC
};
static_assert(sizeof(ClockSnapshot) == 0xD0);

static const u8 kSource[16] = {0x54, 0x6f, 0x74, 0x4b, 0x52, 0x65, 0x63, 0x6f, 0x6d, 0x70, 0x53, 0x74, 0x65, 0x61, 0x64, 0x79};
static s64 now_posix() { return (s64)_time64(nullptr); }
static s64 steady_seconds() { return (s64)(GetTickCount64() / 1000); }
static SteadyClockTimePoint steady_now() { SteadyClockTimePoint p = {steady_seconds(), {}}; memcpy(p.source_id, kSource, 16); return p; }

static void to_calendar(s64 t, bool utc, CalendarTime* ct, CalendarAdditionalInfo* ai) {
    __time64_t tt = (__time64_t)t;
    struct tm tm = {};
    if (utc) _gmtime64_s(&tm, &tt); else _localtime64_s(&tm, &tt);
    if (ct) *ct = {(s16)(tm.tm_year + 1900), (s8)(tm.tm_mon + 1), (s8)tm.tm_mday, (s8)tm.tm_hour, (s8)tm.tm_min, (s8)tm.tm_sec, 0};
    if (ai) {
        memset(ai, 0, sizeof *ai);
        ai->day_of_week = (u32)tm.tm_wday; ai->day_of_year = (u32)tm.tm_yday;
        ai->is_dst = utc ? 0 : tm.tm_isdst > 0;
        struct tm g = {}; _gmtime64_s(&g, &tt);
        ai->utc_offset = utc ? 0 : (s32)(_mkgmtime64(&tm) - _mkgmtime64(&g));
        if (utc) strcpy(ai->tz_name, "UTC");
        else snprintf(ai->tz_name, sizeof ai->tz_name, "%+03d", ai->utc_offset / 3600);
    }
}

HLE_AS(time_Initialize, "_ZN2nn4time10InitializeEv") { c->x[0] = 0; }
HLE_AS(time_Finalize, "_ZN2nn4time8FinalizeEv") { c->x[0] = 0; }
HLE_AS(time_IsInitialized, "_ZN2nn4time13IsInitializedEv") { c->x[0] = 1; }
HLE_AS(time_UserGetCurrentTime, "_ZN2nn4time23StandardUserSystemClock14GetCurrentTimeEPNS0_9PosixTimeE") {
    *(s64*)c->x[0] = now_posix(); c->x[0] = 0;
}
HLE_AS(time_NetworkGetCurrentTime, "_ZN2nn4time26StandardNetworkSystemClock14GetCurrentTimeEPNS0_9PosixTimeE") {
    *(s64*)c->x[0] = now_posix(); c->x[0] = 0;
}
// SteadyClockTimePoint (0x18 bytes) is returned through x8 (AAPCS64 indirect result)
HLE_AS(time_SteadyNow, "_ZN2nn4time19StandardSteadyClock3nowEv") { *(SteadyClockTimePoint*)c->x[8] = steady_now(); }
HLE_AS(time_ToCalendarTime, "_ZN2nn4time14ToCalendarTimeEPNS0_12CalendarTimeEPNS0_22CalendarAdditionalInfoERKNS0_9PosixTimeE") {
    to_calendar(*(s64*)c->x[2], false, (CalendarTime*)c->x[0], (CalendarAdditionalInfo*)c->x[1]); c->x[0] = 0;
}
// CalendarTime (8 bytes) comes back in x0
HLE_AS(time_ToCalendarTimeInUtc, "_ZN2nn4time19ToCalendarTimeInUtcERKNS0_9PosixTimeE") {
    CalendarTime ct; to_calendar(*(s64*)c->x[0], true, &ct, nullptr); u64 r; memcpy(&r, &ct, 8); c->x[0] = r;
}
HLE_AS(time_ToPosixTimeFromUtc, "_ZN2nn4time18ToPosixTimeFromUtcERKNS0_12CalendarTimeE") {
    const CalendarTime* ct = (const CalendarTime*)c->x[0];
    struct tm tm = {};
    tm.tm_year = ct->year - 1900; tm.tm_mon = ct->month - 1; tm.tm_mday = ct->day;
    tm.tm_hour = ct->hour; tm.tm_min = ct->minute; tm.tm_sec = ct->second;
    c->x[0] = (u64)(s64)_mkgmtime64(&tm);
}
HLE_AS(time_ClockSnapshotCtor, "_ZN2nn4time13ClockSnapshotC1Ev") { memset((void*)c->x[0], 0, sizeof(ClockSnapshot)); }
HLE_AS(time_ClockSnapshotCreate, "_ZN2nn4time13ClockSnapshot29CreateWithStandardSystemClockEPS1_") {
    ClockSnapshot* s = (ClockSnapshot*)c->x[0];
    memset(s, 0, sizeof *s);
    s->steady = steady_now();
    s64 t = now_posix();
    s->user_time = s->network_time = t;
    static const s64 offset = now_posix() - steady_seconds();   // fixed per run: "difference by user" stays 0
    s->user_context.offset = s->network_context.offset = offset;
    s->user_context.steady = s->network_context.steady = s->steady;
    to_calendar(t, false, &s->user_cal, &s->user_info);
    to_calendar(t, false, &s->network_cal, &s->network_info);
    strcpy(s->location_name, "UTC");
    s->automatic_correction = 1;
    c->x[0] = 0;
}
// span between two snapshots from the steady clock (same source: always, in one run)
HLE_AS(time_CalculateSpanBetween, "_ZN2nn4time20CalculateSpanBetweenEPNS_8TimeSpanERKNS0_13ClockSnapshotES5_") {
    const ClockSnapshot *a = (const ClockSnapshot*)c->x[1], *b = (const ClockSnapshot*)c->x[2];
    s64 secs = memcmp(a->steady.source_id, b->steady.source_id, 16) == 0 ? b->steady.time_point - a->steady.time_point
                                                                         : b->network_time - a->network_time;
    *(s64*)c->x[0] = secs * 1000000000LL;
    c->x[0] = 0;
}
// the user never changes the clock here: 0 ns (TimeSpan in x0)
HLE_AS(time_UserClockDiffByUser, "_ZN2nn4time48CalculateStandardUserSystemClockDifferenceByUserERKNS0_13ClockSnapshotES3_") {
    const ClockSnapshot *a = (const ClockSnapshot*)c->x[0], *b = (const ClockSnapshot*)c->x[1];
    c->x[0] = (u64)((b->user_context.offset - a->user_context.offset) * 1000000000LL);
}
