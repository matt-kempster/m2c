extern ? a;
extern ? b;

void test(s32 arg0, s32 arg1) {
    ? *var_v0;
    ? *var_v1;
    s32 temp_v0;
    s32 var_a0;
    s32 var_a1;

    var_a0 = arg0;
    var_a1 = arg1;
    var_v1 = &a;
    var_v0 = &b;
    do {
        var_v1->unk0 = (s32) var_v0->unk0;
        var_v1->unk4 = (s32) var_v0->unk4;
        var_v1->unk8 = (s32) var_v0->unk8;
        var_v1->unkC = (s32) var_v0->unkC;
        var_v0 += 0x10;
        var_v1 += 0x10;
    } while (var_v0 != (&b + 0x190));
    temp_v0 = var_a1 + 0x60;
    if ((var_a1 | var_a0) & 3) {
        do {
            var_a0->unk0 = (unaligned s32) var_a1->unk0;
            var_a0->unk4 = (unaligned s32) var_a1->unk4;
            var_a0->unk8 = (unaligned s32) var_a1->unk8;
            var_a0->unkC = (unaligned s32) var_a1->unkC;
            var_a1 += 0x10;
            var_a0 += 0x10;
        } while (var_a1 != temp_v0);
    } else {
        do {
            var_a0->unk0 = (s32) var_a1->unk0;
            var_a0->unk4 = (s32) var_a1->unk4;
            var_a0->unk8 = (s32) var_a1->unk8;
            var_a0->unkC = (s32) var_a1->unkC;
            var_a1 += 0x10;
            var_a0 += 0x10;
        } while (var_a1 != temp_v0);
    }
    *var_a0 = (unaligned s32) var_a1->unk0;
}
