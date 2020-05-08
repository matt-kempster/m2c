s32 test(void) {
    symbol.unk0 = (s32) symbol.unk4;
    return ((symbol.unk0 + 4) + (&symbol + 8)->unk4) + symbol.unk4;
}
