void test(u32 a) {
    test_s8((s8) a);
    test_s16((s16) a);
    test_s32_div((s32) a);
    test_s32_mod((s32) a);
    test_u32_div(a);
    test_u32_mod(a);
}

void test_s8(s8 c) {
    s32 temp_edx;
    s32 temp_edx_2;
    s32 var_edx;

    foo((u32) ((s32) (c - (c >> 0x1F)) >> 1));
    foo(c / 3);
    foo(c / 5);
    temp_edx = (s32) (MULT_HI(-0x6DB6DB6D, c) + c) >> 2;
    foo(temp_edx + ((u32) temp_edx >> 0x1F));
    foo(c / 10);
    foo(c / 100);
    temp_edx_2 = (s32) (MULT_HI(-0x7F7F7F7F, c) + c) >> 7;
    foo(temp_edx_2 + ((u32) temp_edx_2 >> 0x1F));
    var_edx = c & ~0x7FFFFFFE;
    if (var_edx < 0) {
        var_edx = ((var_edx - 1) | ~1) + 1;
    }
    foo((u32) var_edx);
    foo((u32) (c % 3));
    foo((u32) (c % 5));
    foo((u32) (c % 7));
    foo((u32) (c % 10));
    foo((u32) (c % 100));
    foo((u32) (c % 255));
}

void test_s16(s16 h) {
    s32 temp_edx;
    s32 temp_edx_2;
    s32 temp_edx_3;
    s32 temp_edx_4;
    s32 var_edx;

    foo((u32) ((s32) (h - (h >> 0x1F)) >> 1));
    foo(h / 3);
    foo(h / 5);
    temp_edx = (s32) (MULT_HI(-0x6DB6DB6D, h) + h) >> 2;
    foo(temp_edx + ((u32) temp_edx >> 0x1F));
    foo(h / 10);
    foo(h / 100);
    temp_edx_2 = (s32) (MULT_HI(-0x7F7F7F7F, h) + h) >> 7;
    foo(temp_edx_2 + ((u32) temp_edx_2 >> 0x1F));
    temp_edx_3 = (s32) (MULT_HI(-0x49F49F49, h) + h) >> 8;
    foo(temp_edx_3 + ((u32) temp_edx_3 >> 0x1F));
    temp_edx_4 = (s32) (MULT_HI(-0x7FFEFFFD, h) + h) >> 0xF;
    foo(temp_edx_4 + ((u32) temp_edx_4 >> 0x1F));
    var_edx = h & ~0x7FFFFFFE;
    if (var_edx < 0) {
        var_edx = ((var_edx - 1) | ~1) + 1;
    }
    foo((u32) var_edx);
    foo((u32) (h % 3));
    foo((u32) (h % 5));
    foo((u32) (h % 7));
    foo((u32) (h % 10));
    foo((u32) (h % 100));
    foo((u32) (h % 255));
    foo((u32) (h % 360));
    foo((u32) (h % 65534));
}

void test_s32_div(s32 d) {
    s32 temp_edx;
    s32 temp_edx_10;
    s32 temp_edx_11;
    s32 temp_edx_12;
    s32 temp_edx_15;
    s32 temp_edx_17;
    s32 temp_edx_18;
    s32 temp_edx_19;
    s32 temp_edx_20;
    s32 temp_edx_21;
    s32 temp_edx_22;
    s32 temp_edx_2;
    s32 temp_edx_3;
    s32 temp_edx_4;
    s32 temp_edx_5;
    s32 temp_edx_6;
    s32 temp_edx_7;
    s32 temp_edx_8;
    s32 temp_edx_9;
    u32 temp_edx_13;
    u32 temp_edx_14;
    u32 temp_edx_16;

    foo((u32) d);
    foo((u32) ((s32) (d - (d >> 0x1F)) >> 1));
    foo(d / 3);
    foo((u32) ((s32) (d + ((d >> 0x1F) & 3)) >> 2));
    foo(d / 5);
    foo(d / 6);
    temp_edx = (s32) (MULT_HI(-0x6DB6DB6D, d) + d) >> 2;
    foo(temp_edx + ((u32) temp_edx >> 0x1F));
    foo((u32) ((s32) (d + ((d >> 0x1F) & 7)) >> 3));
    foo(d / 9);
    foo(d / 10);
    foo(d / 11);
    foo(d / 12);
    foo(d / 13);
    temp_edx_2 = (s32) (MULT_HI(-0x6DB6DB6D, d) + d) >> 3;
    foo(temp_edx_2 + ((u32) temp_edx_2 >> 0x1F));
    temp_edx_3 = (s32) (MULT_HI(-0x77777777, d) + d) >> 3;
    foo(temp_edx_3 + ((u32) temp_edx_3 >> 0x1F));
    foo((u32) ((s32) (d + ((d >> 0x1F) & 0xF)) >> 4));
    foo(d / 17);
    foo(d / 18);
    foo(d / 19);
    foo(d / 20);
    foo(d / 21);
    foo(d / 22);
    temp_edx_4 = (s32) (MULT_HI(-0x4DE9BD37, d) + d) >> 4;
    foo(temp_edx_4 + ((u32) temp_edx_4 >> 0x1F));
    foo(d / 24);
    foo(d / 25);
    foo(d / 26);
    foo(d / 27);
    temp_edx_5 = (s32) (MULT_HI(-0x6DB6DB6D, d) + d) >> 4;
    foo(temp_edx_5 + ((u32) temp_edx_5 >> 0x1F));
    temp_edx_6 = (s32) (MULT_HI(-0x72C234F7, d) + d) >> 4;
    foo(temp_edx_6 + ((u32) temp_edx_6 >> 0x1F));
    temp_edx_7 = (s32) (MULT_HI(-0x77777777, d) + d) >> 4;
    foo(temp_edx_7 + ((u32) temp_edx_7 >> 0x1F));
    temp_edx_8 = (s32) (MULT_HI(-0x7BDEF7BD, d) + d) >> 4;
    foo(temp_edx_8 + ((u32) temp_edx_8 >> 0x1F));
    foo((u32) ((s32) (d + ((d >> 0x1F) & 0x1F)) >> 5));
    foo(d / 33);
    foo(d / 100);
    temp_edx_9 = (s32) (MULT_HI(-0x7F7F7F7F, d) + d) >> 7;
    foo(temp_edx_9 + ((u32) temp_edx_9 >> 0x1F));
    temp_edx_10 = (s32) (MULT_HI(-0x49F49F49, d) + d) >> 8;
    foo(temp_edx_10 + ((u32) temp_edx_10 >> 0x1F));
    foo(d / 1000);
    foo(d / 10000);
    foo(d / 100000);
    foo(d / 1000000);
    foo(d / 10000000);
    foo(d / 100000000);
    temp_edx_11 = (s32) (MULT_HI(-0x7FFFFFFB, d) + d) >> 0x1D;
    foo(temp_edx_11 + ((u32) temp_edx_11 >> 0x1F));
    temp_edx_12 = (s32) (MULT_HI(-0x7FFFFFFD, d) + d) >> 0x1D;
    foo(temp_edx_12 + ((u32) temp_edx_12 >> 0x1F));
    foo((u32) ((s32) (d + ((d >> 0x1F) & 0x3FFFFFFF)) >> 0x1E));
    temp_edx_13 = d / 1073741824;
    foo(temp_edx_13 + (temp_edx_13 >> 0x1F));
    temp_edx_14 = d / 2147483648;
    foo(temp_edx_14 + (temp_edx_14 >> 0x1F));
    temp_edx_15 = (s32) (MULT_HI(-0x7FFFFFFD, d) + d) >> 0x1E;
    foo(temp_edx_15 + ((u32) temp_edx_15 >> 0x1F));
    temp_edx_16 = d / 2147483648;
    foo(temp_edx_16 + (temp_edx_16 >> 0x1F));
    foo((u32) d >> 0x1F);
    temp_edx_17 = (s32) MULT_HI(-0x40000001, d) >> 0x1D;
    foo(temp_edx_17 + ((u32) temp_edx_17 >> 0x1F));
    temp_edx_18 = (s32) (MULT_HI(0x7FFFFFFD, d) - d) >> 0x1E;
    foo(temp_edx_18 + ((u32) temp_edx_18 >> 0x1F));
    temp_edx_19 = (s32) MULT_HI(-0x66666667, d) >> 2;
    foo(temp_edx_19 + ((u32) temp_edx_19 >> 0x1F));
    temp_edx_20 = (s32) (MULT_HI(0x6DB6DB6D, d) - d) >> 2;
    foo(temp_edx_20 + ((u32) temp_edx_20 >> 0x1F));
    temp_edx_21 = (s32) MULT_HI(-0x66666667, d) >> 1;
    foo(temp_edx_21 + ((u32) temp_edx_21 >> 0x1F));
    foo((u32) -((s32) (d + ((d >> 0x1F) & 3)) >> 2));
    temp_edx_22 = (s32) ((d / 3) - d) >> 1;
    foo(temp_edx_22 + ((u32) temp_edx_22 >> 0x1F));
    foo((u32) -((s32) (d - (d >> 0x1F)) >> 1));
    foo((u32) -d);
}

void test_s32_mod(s32 d) {
    s32 sp40;
    s32 temp_edx;
    s32 temp_edx_2;
    s32 temp_edx_3;
    s32 var_ecx;
    s32 var_edx;
    s32 var_edx_2;
    s32 var_edx_3;
    s32 var_edx_4;
    s32 var_edx_5;

    sp40 = d;
    foo(0U);
    var_ecx = d & ~0x7FFFFFFE;
    if (var_ecx < 0) {
        var_ecx = ((var_ecx - 1) | ~1) + 1;
    }
    foo((u32) var_ecx);
    foo((u32) (d % 3));
    var_edx = d & ~0x7FFFFFFC;
    if (var_edx < 0) {
        var_edx = ((var_edx - 1) | ~3) + 1;
    }
    foo((u32) var_edx);
    foo((u32) (d % 5));
    foo((u32) (d % 6));
    foo((u32) (d % 7));
    var_edx_2 = d & ~0x7FFFFFF8;
    if (var_edx_2 < 0) {
        var_edx_2 = ((var_edx_2 - 1) | ~7) + 1;
    }
    foo((u32) var_edx_2);
    foo((u32) (d % 9));
    foo((u32) (d % 10));
    foo((u32) (d % 11));
    foo((u32) (d % 12));
    foo((u32) (d % 13));
    foo((u32) (d % 14));
    foo((u32) (d % 15));
    var_edx_3 = d & ~0x7FFFFFF0;
    if (var_edx_3 < 0) {
        var_edx_3 = ((var_edx_3 - 1) | ~0xF) + 1;
    }
    foo((u32) var_edx_3);
    foo((u32) (d % 17));
    foo((u32) (d % 18));
    foo((u32) (d % 19));
    foo((u32) (d % 20));
    foo((u32) (d % 21));
    foo((u32) (d % 22));
    foo((u32) (d % 23));
    foo((u32) (d % 24));
    foo((u32) (d % 25));
    foo((u32) (d % 26));
    foo((u32) (d % 27));
    foo((u32) (d % 28));
    foo((u32) (d % 29));
    foo((u32) (d % 30));
    foo((u32) (d % 31));
    var_edx_4 = d & ~0x7FFFFFE0;
    if (var_edx_4 < 0) {
        var_edx_4 = ((var_edx_4 - 1) | ~0x1F) + 1;
    }
    foo((u32) var_edx_4);
    foo((u32) (d % 33));
    foo((u32) (d % 100));
    foo((u32) (d % 255));
    foo((u32) (d % 360));
    foo((u32) (d % 1000));
    foo((u32) (d % 10000));
    foo((u32) (d % 100000));
    foo((u32) (d % 1000000));
    foo((u32) (d % 10000000));
    foo((u32) (d % 100000000));
    foo((u32) (d % 1073741822));
    foo((u32) (d % 1073741823));
    var_edx_5 = d & ~0x40000000;
    if (var_edx_5 < 0) {
        var_edx_5 = ((var_edx_5 - 1) | ~0x3FFFFFFF) + 1;
    }
    foo((u32) var_edx_5);
    foo((u32) (d % 1073741825));
    foo((u32) (d % 2147483645));
    foo((u32) (d % 2147483646));
    foo((u32) (d % 2147483647));
    foo(d & 0x7FFFFFFF);
    foo((u32) (d % -2147483647));
    foo((u32) (d % -2147483646));
    foo((u32) (d % -10));
    foo((u32) (d % -7));
    foo((u32) (d % -5));
    temp_edx = d >> 0x1F;
    foo(((((d ^ temp_edx) - temp_edx) & 3) ^ temp_edx) - temp_edx);
    foo((u32) (d % -3));
    temp_edx_2 = d >> 0x1F;
    foo(((((d ^ temp_edx_2) - temp_edx_2) & 1) ^ temp_edx_2) - temp_edx_2);
    temp_edx_3 = d >> 0x1F;
    foo((0 ^ temp_edx_3) - temp_edx_3);
}

void test_u32_div(u32 u) {
    s32 temp_edx;
    s32 temp_edx_10;
    s32 temp_edx_11;
    s32 temp_edx_12;
    s32 temp_edx_2;
    s32 temp_edx_3;
    s32 temp_edx_4;
    s32 temp_edx_5;
    s32 temp_edx_6;
    s32 temp_edx_7;
    s32 temp_edx_8;
    s32 temp_edx_9;

    foo(u);
    foo(u >> 1);
    foo((u32) MULTU_HI(-0x55555555, u) >> 1);
    foo(u >> 2);
    foo((u32) MULTU_HI(-0x33333333, u) >> 2);
    foo((u32) MULTU_HI(-0x55555555, u) >> 2);
    temp_edx = u / 7;
    foo((u32) (((u32) (u - temp_edx) >> 1) + temp_edx) >> 2);
    foo(u >> 3);
    foo((u32) MULTU_HI(0x38E38E39, u) >> 1);
    foo((u32) MULTU_HI(-0x33333333, u) >> 3);
    foo((u32) MULTU_HI(-0x45D1745D, u) >> 3);
    foo((u32) MULTU_HI(-0x55555555, u) >> 3);
    foo((u32) MULTU_HI(0x4EC4EC4F, u) >> 2);
    temp_edx_2 = u / 7;
    foo((u32) (((u32) (u - temp_edx_2) >> 1) + temp_edx_2) >> 3);
    foo((u32) MULTU_HI(-0x77777777, u) >> 3);
    foo(u >> 4);
    foo((u32) MULTU_HI(-0x0F0F0F0F, u) >> 4);
    foo((u32) MULTU_HI(0x38E38E39, u) >> 2);
    temp_edx_3 = MULTU_HI(-0x50D79435, u);
    foo((u32) (((u32) (u - temp_edx_3) >> 1) + temp_edx_3) >> 4);
    foo((u32) MULTU_HI(-0x33333333, u) >> 4);
    temp_edx_4 = MULTU_HI(-0x79E79E79, u);
    foo((u32) (((u32) (u - temp_edx_4) >> 1) + temp_edx_4) >> 4);
    foo((u32) MULTU_HI(-0x45D1745D, u) >> 4);
    foo((u32) MULTU_HI(-0x4DE9BD37, u) >> 4);
    foo((u32) MULTU_HI(-0x55555555, u) >> 4);
    foo((u32) MULTU_HI(0x51EB851F, u) >> 3);
    foo((u32) MULTU_HI(0x4EC4EC4F, u) >> 3);
    temp_edx_5 = MULTU_HI(0x2F684BDB, u);
    foo((u32) (((u32) (u - temp_edx_5) >> 1) + temp_edx_5) >> 4);
    temp_edx_6 = u / 7;
    foo((u32) (((u32) (u - temp_edx_6) >> 1) + temp_edx_6) >> 4);
    foo((u32) MULTU_HI(-0x72C234F7, u) >> 4);
    foo((u32) MULTU_HI(-0x77777777, u) >> 4);
    temp_edx_7 = u / 31;
    foo((u32) (((u32) (u - temp_edx_7) >> 1) + temp_edx_7) >> 4);
    foo(u >> 5);
    foo((u32) MULTU_HI(0x3E0F83E1, u) >> 3);
    foo((u32) MULTU_HI(0x51EB851F, u) >> 5);
    foo((u32) MULTU_HI(-0x7F7F7F7F, u) >> 7);
    temp_edx_8 = MULTU_HI(0x6C16C16D, u);
    foo((u32) (((u32) (u - temp_edx_8) >> 1) + temp_edx_8) >> 8);
    foo((u32) MULTU_HI(0x10624DD3, u) >> 6);
    foo((u32) MULTU_HI(-0x2E48E8A7, u) >> 0xD);
    temp_edx_9 = MULTU_HI(0x4F8B588F, u);
    foo((u32) (((u32) (u - temp_edx_9) >> 1) + temp_edx_9) >> 0x10);
    foo((u32) MULTU_HI(0x431BDE83, u) >> 0x12);
    foo((u32) MULTU_HI(0x6B5FCA6B, u) >> 0x16);
    foo((u32) MULTU_HI(0x55E63B89, u) >> 0x19);
    foo(u >> 0x1E);
    foo((u32) MULTU_HI(-3, u) >> 0x1E);
    temp_edx_10 = u / 858993459;
    foo((u32) (((u32) (u - temp_edx_10) >> 1) + temp_edx_10) >> 0x1E);
    temp_edx_11 = u / 1431655765;
    foo((u32) (((u32) (u - temp_edx_11) >> 1) + temp_edx_11) >> 0x1E);
    foo(u >> 0x1F);
    temp_edx_12 = MULTU_HI(-3, u);
    foo((u32) (((u32) (u - temp_edx_12) >> 1) + temp_edx_12) >> 0x1F);
    foo(u / -2U);
    foo(u / -1U);
}

void test_u32_mod(u32 u) {
    u32 sp40;

    sp40 = u;
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
    foo(u % 100000U);
    foo(u % 1000000U);
    foo(u % 10000000U);
    foo(u % 100000000U);
    foo(u & 0x3FFFFFFF);
    foo(u % 1073741825U);
    foo(u % 2147483646U);
    foo(u % 2147483647U);
    foo(u & 0x7FFFFFFF);
    foo(u % -2147483647U);
    foo(u % -2U);
    foo(u);
}
