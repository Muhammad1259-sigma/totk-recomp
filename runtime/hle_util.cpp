// HLE: nn::util binary file helpers (NintendoWare BinaryFileHeader / BinaryBlockHeader / RelocationTable).
// Semantics copied from the SDK's own code (sdk image 0x7d5a0-0x7dc38, disassembled 2026-10-14), not from docs:
//   file header: +0 u64 signature, +9 u8 minor, +0xA u16 major, +0xC s16 BOM (0xFEFF), +0xE u8 alignment log2,
//                +0x14 u16 flags (bit0 relocated), +0x16 u16 first block, +0x18 u32 relocation table, +0x1C u32 size
//   block header: +4 u32 next block offset, +8 u32 block size
//   relocation table: +4 u32 position in file, +8 s32 section count, sections at +0x10 (0x18 each:
//                +0 u64 ptr, +8 u32 position, +0xC u32 size, +0x10 s32 entry index, +0x14 s32 entry count),
//                entries after the sections (8 each: u32 position, u16 struct count, u8 offset count, u8 padding).
#include "hle.h"

static u8* off_or_null(u8* p, u32 off) { return off ? p + off : nullptr; }

HLE_AS(util_BFH_IsSignatureValid, "_ZNK2nn4util16BinaryFileHeader16IsSignatureValidEl") { c->x[0] = *(u64*)c->x[0] == c->x[1]; }
HLE_AS(util_BFH_IsAlignmentValid, "_ZNK2nn4util16BinaryFileHeader16IsAlignmentValidEv") {
    u8* h = (u8*)c->x[0]; c->x[0] = ((u64)h & ~(~0ull << h[0xE])) == 0;
}
HLE_AS(util_BFH_GetAlignment, "_ZNK2nn4util16BinaryFileHeader12GetAlignmentEv") { c->x[0] = 1ull << ((u8*)c->x[0])[0xE]; }
HLE_AS(util_BFH_IsEndianReverse, "_ZNK2nn4util16BinaryFileHeader15IsEndianReverseEv") { c->x[0] = *(s16*)(c->x[0] + 0xC) == -2; }
HLE_AS(util_BFH_IsRelocated, "_ZNK2nn4util16BinaryFileHeader11IsRelocatedEv") { c->x[0] = *(u16*)(c->x[0] + 0x14) & 1; }
HLE_AS(util_BFH_GetFileSize, "_ZNK2nn4util16BinaryFileHeader11GetFileSizeEv") { c->x[0] = *(u32*)(c->x[0] + 0x1C); }
HLE_AS(util_BFH_GetRelocationTable, "_ZN2nn4util16BinaryFileHeader18GetRelocationTableEv") {
    u8* h = (u8*)c->x[0]; c->x[0] = (u64)off_or_null(h, *(u32*)(h + 0x18));
}
HLE_AS(util_BFH_GetFirstBlock, "_ZN2nn4util16BinaryFileHeader13GetFirstBlockEv") {
    u8* h = (u8*)c->x[0]; c->x[0] = (u64)off_or_null(h, *(u16*)(h + 0x16));
}
HLE_AS(util_BFH_GetFirstBlockC, "_ZNK2nn4util16BinaryFileHeader13GetFirstBlockEv") {
    u8* h = (u8*)c->x[0]; c->x[0] = (u64)off_or_null(h, *(u16*)(h + 0x16));
}
// IsValid(signature, major, minor, micro): micro is not checked by the SDK either
HLE_AS(util_BFH_IsValid, "_ZNK2nn4util16BinaryFileHeader7IsValidEliii") {
    u8* h = (u8*)c->x[0];
    bool ok = *(u64*)h == c->x[1] && *(u16*)(h + 0xA) == (u32)c->x[2] && (s32)h[9] <= (s32)c->x[3] &&
              *(s16*)(h + 0xC) == (s16)0xFEFF && ((u64)h & ~(~0ull << h[0xE])) == 0;
    c->x[0] = ok;
}
HLE_AS(util_BBH_GetBlockSize, "_ZNK2nn4util17BinaryBlockHeader12GetBlockSizeEv") { c->x[0] = *(u32*)(c->x[0] + 8); }
HLE_AS(util_BBH_GetNextBlockC, "_ZNK2nn4util17BinaryBlockHeader12GetNextBlockEv") {
    u8* b = (u8*)c->x[0]; c->x[0] = (u64)off_or_null(b, *(u32*)(b + 4));
}

static void relocate(u8* t, bool undo) {
    u8* file = t - *(u32*)(t + 4);
    s32 nsec = *(s32*)(t + 8);
    u8* entries = t + 0x10 + (u64)nsec * 0x18;
    for (s32 i = 0; i < nsec; i++) {
        u8* sec = t + 0x10 + (u64)i * 0x18;
        u64 ptr = *(u64*)sec;
        u64 base = ptr ? ptr - *(u32*)(sec + 8) : (u64)file;
        if (undo) *(u64*)sec = 0;
        s32 e0 = *(s32*)(sec + 0x10), ne = *(s32*)(sec + 0x14);
        for (s32 e = e0; e < e0 + ne; e++) {
            u8* en = entries + (u64)e * 8;
            u16 nstruct = *(u16*)(en + 4); u8 noff = en[6], pad = en[7];
            if (!nstruct || !noff) continue;
            u64* p = (u64*)(file + *(u32*)en);
            for (u32 s = 0; s < nstruct; s++, p += pad)
                for (u32 o = 0; o < noff; o++, p++)
                    if (*p) *p = undo ? *p - base : *p + base;
        }
    }
    if (undo) *(u16*)(file + 0x14) &= ~1; else *(u16*)(file + 0x14) |= 1;
}
HLE_AS(util_RT_Relocate, "_ZN2nn4util15RelocationTable8RelocateEv") { relocate((u8*)c->x[0], false); }
HLE_AS(util_RT_Unrelocate, "_ZN2nn4util15RelocationTable10UnrelocateEv") { relocate((u8*)c->x[0], true); }
