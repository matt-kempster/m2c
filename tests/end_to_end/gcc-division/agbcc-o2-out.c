void test(u32 a) {
    test_s8((s8) (u8) a);
    test_s16((s16) a);
    test_s32_div((s32) a);
    test_s32_mod((s32) a);
    test_u32_div(a);
    test_u32_mod(a);
}

void test_s8(s8 c) {
    unksp0 = c;
    foo((u8) unksp0 >> 1);
    foo((u32) (u8) ((u8) unksp0 / 3U));
    foo((u32) (u8) ((u8) unksp0 / 5U));
    foo((u32) (u8) ((u8) unksp0 / 7U));
    foo((u32) (u8) ((u8) unksp0 / 10U));
    foo((u32) (u8) ((u8) unksp0 / 100U));
    foo((u32) (u8) ((u8) unksp0 / 255U));
    foo(1 & (u8) unksp0);
    foo((u32) (u8) ((u8) unksp0 % 3U));
    foo((u32) (u8) ((u8) unksp0 % 5U));
    foo((u32) (u8) ((u8) unksp0 % 7U));
    foo((u32) (u8) ((u8) unksp0 % 10U));
    foo((u32) (u8) ((u8) unksp0 % 100U));
    foo((u32) (u8) ((u8) unksp0 % 255U));
}

void test_s16(s16 h) {
    s16 temp_r0;
    u16 temp_r1;
    u16 temp_r1_2;

    unksp0 = h;
    temp_r1 = (u16) unksp0;
    foo((u32) ((s32) ((s16) temp_r1 + ((u32) (temp_r1 << 0x10) >> 0x1F)) >> 1));
    foo((u32) (s16) ((s16) (u16) unksp0 / 3));
    foo((u32) (s16) ((s16) (u16) unksp0 / 5));
    foo((u32) (s16) ((s16) (u16) unksp0 / 7));
    foo((u32) (s16) ((s16) (u16) unksp0 / 10));
    foo((u32) (s16) ((s16) (u16) unksp0 / 100));
    foo((u32) (s16) ((s16) (u16) unksp0 / 255));
    foo((u32) (s16) ((s16) (u16) unksp0 / 360));
    foo((u32) ((s16) (u16) unksp0 / 65534));
    temp_r1_2 = (u16) unksp0;
    temp_r0 = (s16) temp_r1_2;
    foo((u32) (s16) (temp_r0 - (((s32) (temp_r0 + ((u32) (temp_r1_2 << 0x10) >> 0x1F)) >> 1) * 2)));
    foo((u32) (s16) ((s16) (u16) unksp0 % 3));
    foo((u32) (s16) ((s16) (u16) unksp0 % 5));
    foo((u32) (s16) ((s16) (u16) unksp0 % 7));
    foo((u32) (s16) ((s16) (u16) unksp0 % 10));
    foo((u32) (s16) ((s16) (u16) unksp0 % 100));
    foo((u32) (s16) ((s16) (u16) unksp0 % 255));
    foo((u32) (s16) ((s16) (u16) unksp0 % 360));
    foo((u32) ((s16) (u16) unksp0 % 65534));
}

void test_s32_div(s32 d) {
    s32 var_r0;
    s32 var_r0_2;
    s32 var_r0_3;
    s32 var_r0_4;
    s32 var_r0_5;
    s32 var_r0_6;

    unksp0 = d;
    foo((u32) d);
    foo((u32) ((s32) (d + ((u32) d >> 0x1F)) >> 1));
    foo((u32) (d / 3));
    var_r0 = d;
    if (var_r0 < 0) {
        var_r0 += 3;
    }
    foo((u32) (var_r0 >> 2));
    foo((u32) (unksp0 / 5));
    foo((u32) (unksp0 / 6));
    foo((u32) (unksp0 / 7));
    var_r0_2 = unksp0;
    if (var_r0_2 < 0) {
        var_r0_2 += 7;
    }
    foo((u32) (var_r0_2 >> 3));
    foo((u32) (unksp0 / 9));
    foo((u32) (unksp0 / 10));
    foo((u32) (unksp0 / 11));
    foo((u32) (unksp0 / 12));
    foo((u32) (unksp0 / 13));
    foo((u32) (unksp0 / 14));
    foo((u32) (unksp0 / 15));
    var_r0_3 = unksp0;
    if (var_r0_3 < 0) {
        var_r0_3 += 0xF;
    }
    foo((u32) (var_r0_3 >> 4));
    foo((u32) (unksp0 / 17));
    foo((u32) (unksp0 / 18));
    foo((u32) (unksp0 / 19));
    foo((u32) (unksp0 / 20));
    foo((u32) (unksp0 / 21));
    foo((u32) (unksp0 / 22));
    foo((u32) (unksp0 / 23));
    foo((u32) (unksp0 / 24));
    foo((u32) (unksp0 / 25));
    foo((u32) (unksp0 / 26));
    foo((u32) (unksp0 / 27));
    foo((u32) (unksp0 / 28));
    foo((u32) (unksp0 / 29));
    foo((u32) (unksp0 / 30));
    foo((u32) (unksp0 / 31));
    var_r0_4 = unksp0;
    if (var_r0_4 < 0) {
        var_r0_4 += 0x1F;
    }
    foo((u32) (var_r0_4 >> 5));
    foo((u32) (unksp0 / 33));
    foo((u32) (unksp0 / 100));
    foo((u32) (unksp0 / 255));
    foo((u32) (unksp0 / 360));
    foo((u32) (unksp0 / 1000));
    foo((u32) (unksp0 / 10000));
    foo((u32) (unksp0 / (s32) .L13.unk4));
    foo((u32) (unksp0 / (s32) .L13.unk8));
    foo((u32) (unksp0 / (s32) .L13.unkC));
    foo((u32) (unksp0 / (s32) .L13.unk10));
    foo((u32) (unksp0 / (s32) .L13.unk14));
    foo((u32) (unksp0 / (s32) .L13.unk18));
    var_r0_5 = unksp0;
    if (var_r0_5 < 0) {
        var_r0_5 += .L13.unk18;
    }
    foo((u32) (var_r0_5 >> 0x1E));
    foo((u32) (unksp0 / (s32) .L13.unk1C));
    foo((u32) (unksp0 / (s32) .L13.unk20));
    foo((u32) (unksp0 / (s32) .L13.unk24));
    foo((u32) (unksp0 / (s32) .L13.unk28));
    foo(unksp0 >> 0x1F);
    foo((u32) (unksp0 / (s32) .L13.unk2C));
    foo((u32) (unksp0 / (s32) .L13.unk30));
    foo((u32) (unksp0 / -10));
    foo((u32) (unksp0 / -7));
    foo((u32) (unksp0 / -5));
    var_r0_6 = unksp0;
    if (var_r0_6 < 0) {
        var_r0_6 += 3;
    }
    foo(0 - (var_r0_6 >> 2));
    foo((u32) (unksp0 / -3));
    foo(0 - ((s32) (unksp0 + (unksp0 >> 0x1F)) >> 1));
    foo(0 - unksp0);
}

void test_s32_mod(s32 d) {
    s32 temp_r4;
    s32 temp_r5;
    s32 var_r0;
    s32 var_r0_2;
    s32 var_r0_3;
    s32 var_r0_4;
    s32 var_r0_5;
    s32 var_r0_6;

    unksp0 = d;
    foo(0U);
    foo(d - (((s32) (d + ((u32) d >> 0x1F)) >> 1) * 2));
    foo((u32) (d % 3));
    var_r0 = unksp0;
    if (unksp0 < 0) {
        var_r0 = unksp0 + 3;
    }
    foo(unksp0 - ((var_r0 >> 2) * 4));
    foo((u32) (unksp0 % 5));
    foo((u32) (unksp0 % 6));
    foo((u32) (unksp0 % 7));
    var_r0_2 = unksp0;
    if (unksp0 < 0) {
        var_r0_2 = unksp0 + 7;
    }
    foo(unksp0 - ((var_r0_2 >> 3) * 8));
    foo((u32) (unksp0 % 9));
    foo((u32) (unksp0 % 10));
    foo((u32) (unksp0 % 11));
    foo((u32) (unksp0 % 12));
    foo((u32) (unksp0 % 13));
    foo((u32) (unksp0 % 14));
    foo((u32) (unksp0 % 15));
    var_r0_3 = unksp0;
    if (unksp0 < 0) {
        var_r0_3 += 0xF;
    }
    foo(unksp0 - ((var_r0_3 >> 4) * 0x10));
    foo((u32) (unksp0 % 17));
    foo((u32) (unksp0 % 18));
    foo((u32) (unksp0 % 19));
    foo((u32) (unksp0 % 20));
    foo((u32) (unksp0 % 21));
    foo((u32) (unksp0 % 22));
    foo((u32) (unksp0 % 23));
    foo((u32) (unksp0 % 24));
    foo((u32) (unksp0 % 25));
    foo((u32) (unksp0 % 26));
    foo((u32) (unksp0 % 27));
    foo((u32) (unksp0 % 28));
    foo((u32) (unksp0 % 29));
    foo((u32) (unksp0 % 30));
    foo((u32) (unksp0 % 31));
    var_r0_4 = unksp0;
    if (unksp0 < 0) {
        var_r0_4 += 0x1F;
    }
    foo(unksp0 - ((var_r0_4 >> 5) << 5));
    foo((u32) (unksp0 % 33));
    foo((u32) (unksp0 % 100));
    foo((u32) (unksp0 % 255));
    foo((u32) (unksp0 % 360));
    foo((u32) (unksp0 % 1000));
    foo((u32) (unksp0 % 10000));
    foo((u32) (unksp0 % (s32) .L22.unk4));
    foo((u32) (unksp0 % (s32) .L22.unk8));
    foo((u32) (unksp0 % (s32) .L22.unkC));
    foo((u32) (unksp0 % (s32) .L22.unk10));
    foo((u32) (unksp0 % (s32) .L22.unk14));
    foo((u32) (unksp0 % (s32) .L22.unk18));
    var_r0_5 = unksp0;
    if (unksp0 < 0) {
        var_r0_5 = unksp0 + .L22.unk18;
    }
    foo(unksp0 - ((var_r0_5 >> 0x1E) << 0x1E));
    foo((u32) (unksp0 % (s32) .L22.unk1C));
    foo((u32) (unksp0 % (s32) .L22.unk20));
    temp_r5 = .L22.unk24;
    foo((u32) (unksp0 % temp_r5));
    temp_r4 = .L22.unk28;
    foo((u32) (unksp0 % temp_r4));
    foo(unksp0 & temp_r4);
    foo((u32) (unksp0 % temp_r4));
    foo((u32) (unksp0 % temp_r5));
    foo((u32) (unksp0 % 10));
    foo((u32) (unksp0 % 7));
    foo((u32) (unksp0 % 5));
    var_r0_6 = unksp0;
    if (unksp0 < 0) {
        var_r0_6 = unksp0 + 3;
    }
    foo(unksp0 - ((var_r0_6 >> 2) * 4));
    foo((u32) (unksp0 % 3));
    foo(unksp0 - (((s32) (unksp0 + (unksp0 >> 0x1F)) >> 1) * 2));
    foo(0U);
}

void test_u32_div(u32 u) {
    unksp0 = u;
    foo(u);
    foo(u >> 1);
    foo(u / 3U);
    foo(u >> 2);
    foo(u / 5U);
    foo(u / 6U);
    foo(u / 7U);
    foo(u >> 3);
    foo(u / 9U);
    foo(u / 10U);
    foo(u / 11U);
    foo(u / 12U);
    foo(u / 13U);
    foo(u / 14U);
    foo(u / 15U);
    foo(u >> 4);
    foo(u / 17U);
    foo(u / 18U);
    foo(u / 19U);
    foo(u / 20U);
    foo(u / 21U);
    foo(u / 22U);
    foo(u / 23U);
    foo(u / 24U);
    foo(u / 25U);
    foo(u / 26U);
    foo(u / 27U);
    foo(u / 28U);
    foo(u / 29U);
    foo(u / 30U);
    foo(u / 31U);
    foo(u >> 5);
    foo(u / 33U);
    foo(u / 100U);
    foo(u / 255U);
    foo(u / 360U);
    foo(u / 1000U);
    foo(u / 10000U);
    foo(u / (u32) .L25.unk4);
    foo(u / (u32) .L25.unk8);
    foo(u / (u32) .L25.unkC);
    foo(u / (u32) .L25.unk10);
    foo(u >> 0x1E);
    foo(u / (u32) .L25.unk14);
    foo(u / (u32) .L25.unk18);
    foo(u / (u32) .L25.unk1C);
    foo(u >> 0x1F);
    foo(u / (u32) .L25.unk20);
    foo(u / -2U);
    foo(u / -1U);
}

void test_u32_mod(u32 u) {
    u32 temp_r4;

    unksp0 = u;
    foo(0U);
    foo(u & 1);
    foo(u % 3U);
    foo(u & 3);
    foo(u % 5U);
    foo(u % 6U);
    foo(u % 7U);
    foo(u & 7);
    foo(u % 9U);
    foo(u % 10U);
    foo(u % 11U);
    foo(u % 12U);
    foo(u % 13U);
    foo(u % 14U);
    foo(u % 15U);
    foo(u & 0xF);
    foo(u % 17U);
    foo(u % 18U);
    foo(u % 19U);
    foo(u % 20U);
    foo(u % 21U);
    foo(u % 22U);
    foo(u % 23U);
    foo(u % 24U);
    foo(u % 25U);
    foo(u % 26U);
    foo(u % 27U);
    foo(u % 28U);
    foo(u % 29U);
    foo(u % 30U);
    foo(u % 31U);
    foo(u & 0x1F);
    foo(u % 33U);
    foo(u % 100U);
    foo(u % 255U);
    foo(u % 360U);
    foo(u % 1000U);
    foo(u % 10000U);
    foo(u % (u32) .L28.unk4);
    foo(u % (u32) .L28.unk8);
    foo(u % (u32) .L28.unkC);
    foo(u % (u32) .L28.unk10);
    foo(u & .L28.unk14);
    foo(u % (u32) .L28.unk18);
    foo(u % (u32) .L28.unk1C);
    temp_r4 = .L28.unk20;
    foo(u % temp_r4);
    foo(u & temp_r4);
    foo(u % (u32) .L28.unk24);
    foo(u % -2U);
    foo(u % -1U);
}
