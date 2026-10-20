// HLE: nn::account with one local user (preselected at launch, like a single-profile console).
#include "hle.h"

static const u64 kUid[2] = { 0x0123456789ABCDEFull, 0x0FEDCBA987654321ull };   // nn::account::Uid {u64,u64}

HLE_AS(acc_Initialize, "_ZN2nn7account10InitializeEv") { }
// UserHandle is {u64 id, u64 ptr, u64 ...}; we only need it to be non-null and recognisable
HLE_AS(acc_TryOpenPreselectedUser, "_ZN2nn7account22TryOpenPreselectedUserEPNS0_10UserHandleE") {
    u64* h = (u64*)c->x[0]; h[0] = kUid[0]; h[1] = kUid[1]; h[2] = 1;
    c->x[0] = 1;
}
HLE_AS(acc_CloseUser, "_ZN2nn7account9CloseUserERKNS0_10UserHandleE") { }
HLE_AS(acc_GetUserId, "_ZN2nn7account9GetUserIdEPNS0_3UidERKNS0_10UserHandleE") {
    u64* u = (u64*)c->x[0]; u[0] = kUid[0]; u[1] = kUid[1]; c->x[0] = 0;
}
HLE_AS(acc_GetUserExistence, "_ZN2nn7account16GetUserExistenceEPbRKNS0_3UidE") { *(u8*)c->x[0] = 1; c->x[0] = 0; }
HLE_AS(acc_GetNickname, "_ZN2nn7account11GetNicknameEPNS0_8NicknameERKNS0_3UidE") {   // Nickname = char[33]
    char* n = (char*)c->x[0]; memset(n, 0, 33); memcpy(n, "Link", 4); c->x[0] = 0;
}
