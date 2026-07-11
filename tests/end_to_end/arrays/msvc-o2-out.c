extern ? ??_C@_05DLON@hello?$AA@;

s32 test(s32 index, s32 *argArray, struct S *s) {
    s32 sp4;
    s16 sp8;

    sp8 = ??_C@_05DLON@hello?$AA@.unk4;
    sp4 = ??_C@_05DLON@hello?$AA@.unk0;
    return (argArray[index] * *(sp + (index + 4))) + s->b[index] + _globalArray[index];
}
