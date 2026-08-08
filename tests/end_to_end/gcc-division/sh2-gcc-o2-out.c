void test(u32 a) {
    test_s8((s8) a);
    test_s16((s16) a);
    test_s32_div((s32) a);
    test_s32_mod((s32) a);
    test_u32_div(a);
    test_u32_mod(a);
}

void test_s8(s8 c) {
    s8 sp0;

    sp0 = c;
    foo((u32) ((s32) (((u32) sp0 >> 0x1F) + sp0) >> 1));
    foo((u32) (s8) (sp0 / 3));
    foo((u32) (s8) (sp0 / 5));
    foo((u32) (s8) (sp0 / 7));
    foo((u32) (s8) (sp0 / 10));
    foo((u32) (s8) (sp0 / 100));
    foo(sp0 / 255);
    foo((u32) (s8) (c - (((s32) (((u32) c >> 0x1F) + c) >> 1) * 2)));
    foo((u32) (s8) (c % 3));
    foo((u32) (s8) (c % 5));
    foo((u32) (s8) (c % 7));
    foo((u32) (s8) (c % 10));
    foo((u32) (s8) (c % 100));
    foo(sp0 % 255);
}

void test_s16(s16 h) {
    s16 sp0;

    sp0 = h;
    foo((u32) ((s32) (((u32) sp0 >> 0x1F) + sp0) >> 1));
    foo((u32) (s16) (sp0 / 3));
    foo((u32) (s16) (sp0 / 5));
    foo((u32) (s16) (sp0 / 7));
    foo((u32) (s16) (sp0 / 10));
    foo((u32) (s16) (sp0 / 100));
    foo((u32) (s16) (sp0 / 255));
    foo((u32) (s16) (sp0 / 360));
    foo(sp0 / 65534);
    foo((u32) (s16) (h - (((s32) (((u32) h >> 0x1F) + h) >> 1) * 2)));
    foo((u32) (s16) (h % 3));
    foo((u32) (s16) (h % 5));
    foo((u32) (s16) (h % 7));
    foo((u32) (s16) (h % 10));
    foo((u32) (s16) (h % 100));
    foo((u32) (s16) (sp0 % 255));
    foo((u32) (s16) (sp0 % 360));
    foo(sp0 % 65534);
}

void test_s32_div(s32 d) {
    s32 sp0;
    s32 var_r4;
    s32 var_r4_2;
    s32 var_r4_3;
    s32 var_r4_4;
    s32 var_r4_5;
    s32 var_r4_6;

    sp0 = d;
    foo((u32) d);
    foo((u32) ((s32) (d + ((u32) d >> 0x1F)) >> 1));
    foo(sp0 / 3);
    var_r4 = d;
    if (var_r4 < 0) {
        var_r4 += 3;
    }
    foo((u32) (var_r4 >> 2));
    foo(sp0 / 5);
    foo(sp0 / 6);
    foo(sp0 / 7);
    var_r4_2 = sp0;
    if (var_r4_2 < 0) {
        var_r4_2 += 7;
    }
    foo((u32) (var_r4_2 >> 3));
    foo(sp0 / 9);
    foo(sp0 / 10);
    foo(sp0 / 11);
    foo(sp0 / 12);
    foo(sp0 / 13);
    foo(sp0 / 14);
    foo(sp0 / 15);
    var_r4_3 = sp0;
    if (var_r4_3 < 0) {
        var_r4_3 += 0xF;
    }
    foo((u32) (var_r4_3 >> 4));
    foo(sp0 / 17);
    foo(sp0 / 18);
    foo(sp0 / 19);
    foo(sp0 / 20);
    foo(sp0 / 21);
    foo(sp0 / 22);
    foo(sp0 / 23);
    foo(sp0 / 24);
    foo(sp0 / 25);
    foo(sp0 / 26);
    foo(sp0 / 27);
    foo(sp0 / 28);
    foo(sp0 / 29);
    foo(sp0 / 30);
    foo(sp0 / 31);
    var_r4_4 = sp0;
    if (var_r4_4 < 0) {
        var_r4_4 += 0x1F;
    }
    foo((u32) (var_r4_4 >> 5));
    foo(sp0 / 33);
    foo(sp0 / 100);
    foo(sp0 / 255);
    foo(sp0 / 360);
    foo(sp0 / 1000);
    foo(sp0 / 10000);
    foo(sp0 / 100000);
    foo(sp0 / 1000000);
    foo(sp0 / 10000000);
    foo(sp0 / 100000000);
    foo(sp0 / 1073741822);
    foo(sp0 / 1073741822);
    var_r4_5 = sp0;
    if (var_r4_5 < 0) {
        var_r4_5 += 0x3FFFFFFF;
    }
    foo((u32) (var_r4_5 >> 0x1E));
    foo(sp0 / 1073741824);
    foo(sp0 / 2147483648);
    foo(sp0 / 2147483645);
    foo(sp0 / 2147483648);
    foo((u32) sp0 >> 0x1F);
    foo(sp0 / -2147483648);
    foo(sp0 / -2147483645);
    foo(sp0 / -10);
    foo(sp0 / -7);
    foo(sp0 / -5);
    var_r4_6 = sp0;
    if (var_r4_6 < 0) {
        var_r4_6 += 3;
    }
    foo((u32) -(var_r4_6 >> 2));
    foo(sp0 / -3);
    foo((u32) -((s32) (sp0 + ((u32) sp0 >> 0x1F)) >> 1));
    foo((u32) -sp0);
}

void test_s32_mod(s32 d) {
    s32 sp0;
    s32 var_r1;
    s32 var_r1_2;
    s32 var_r1_3;
    s32 var_r2;
    s32 var_r2_2;
    s32 var_r4;

    sp0 = d;
    foo(0U);
    foo(d - (((s32) (((u32) d >> 0x1F) + d) >> 1) * 2));
    foo(d % 3);
    var_r2 = d;
    if (d < 0) {
        var_r2 += 3;
    }
    foo(d - (var_r2 & ~3));
    foo(sp0 % 5);
    foo(sp0 % 6);
    foo(sp0 % 7);
    var_r1 = sp0;
    if (sp0 < 0) {
        var_r1 += 7;
    }
    foo(sp0 - ((var_r1 >> 3) * 8));
    foo(sp0 % 9);
    foo(sp0 % 10);
    foo(sp0 % 11);
    foo(sp0 % 12);
    foo(sp0 % 13);
    foo(sp0 % 14);
    foo(sp0 % 15);
    var_r1_2 = sp0;
    if (sp0 < 0) {
        var_r1_2 += 0xF;
    }
    foo(sp0 - ((var_r1_2 >> 4) * 0x10));
    foo(sp0 % 17);
    foo(sp0 % 18);
    foo(sp0 % 19);
    foo(sp0 % 20);
    foo(sp0 % 21);
    foo(sp0 % 22);
    foo(sp0 % 23);
    foo(sp0 % 24);
    foo(sp0 % 25);
    foo(sp0 % 26);
    foo(sp0 % 27);
    foo(sp0 % 28);
    foo(sp0 % 29);
    foo(sp0 % 30);
    foo(sp0 % 31);
    var_r1_3 = sp0;
    if (sp0 < 0) {
        var_r1_3 += 0x1F;
    }
    foo(sp0 - ((var_r1_3 >> 5) << 5));
    foo(sp0 % 33);
    foo(sp0 % 100);
    foo(sp0 % 255);
    foo(sp0 % 360);
    foo(sp0 % 1000);
    foo(sp0 % 10000);
    foo(sp0 % 100000);
    foo(sp0 % 1000000);
    foo(sp0 % 10000000);
    foo(sp0 % 100000000);
    foo(sp0 % 1073741822);
    foo(sp0 - ((sp0 / 1073741822) * 0x3FFFFFFF));
    var_r4 = sp0;
    if (sp0 < 0) {
        var_r4 += 0x3FFFFFFF;
    }
    foo(sp0 - (((u32) ((var_r4 >> 0x1E) << 0x10) >> 2) << 0x10));
    foo(sp0 - ((sp0 / 1073741824) * 0x40000001));
    foo(sp0 - ((sp0 / 2147483648) * 0x7FFFFFFD));
    foo(sp0 - ((sp0 / 2147483645) * 0x7FFFFFFE));
    foo(sp0 - ((sp0 / 2147483648) * 0x7FFFFFFF));
    foo(sp0 & 0x7FFFFFFF);
    foo(sp0 - ((sp0 / 2147483648) * 0x7FFFFFFF));
    foo(sp0 - ((sp0 / 2147483645) * 0x7FFFFFFE));
    foo(sp0 % 10);
    foo(sp0 % 7);
    foo(sp0 % 5);
    var_r2_2 = sp0;
    if (sp0 < 0) {
        var_r2_2 += 3;
    }
    foo(sp0 - (var_r2_2 & ~3));
    foo(sp0 % 3);
    foo(sp0 - (((s32) (((u32) sp0 >> 0x1F) + sp0) >> 1) * 2));
    foo(0U);
}

void test_u32_div(u32 u) {
    u32 sp0;
    s32 temp_mach;
    s32 temp_mach_2;
    s32 temp_mach_3;
    s32 temp_mach_4;
    s32 temp_mach_5;
    s32 temp_mach_6;

    sp0 = u;
    foo(u);
    foo(u >> 1);
    foo(sp0 / 3);
    foo(u >> 2);
    foo(sp0 / 5);
    foo(sp0 / 6);
    temp_mach = sp0 / 7;
    foo((u32) (temp_mach + ((u32) (sp0 - temp_mach) >> 1)) >> 2);
    foo(u >> 3);
    foo(sp0 / 9);
    foo(sp0 / 10);
    foo(sp0 / 11);
    foo(sp0 / 12);
    foo(sp0 / 13);
    foo(sp0 / 14);
    foo(sp0 / 15);
    foo(u >> 4);
    foo(sp0 / 17);
    foo(sp0 / 18);
    temp_mach_2 = MULTU_HI(sp0, 0xAF286BCB);
    foo((u32) (temp_mach_2 + ((u32) (sp0 - temp_mach_2) >> 1)) >> 4);
    foo(sp0 / 20);
    temp_mach_3 = MULTU_HI(sp0, 0x86186187);
    foo((u32) (temp_mach_3 + ((u32) (sp0 - temp_mach_3) >> 1)) >> 4);
    foo(sp0 / 22);
    foo(sp0 / 23);
    foo(sp0 / 24);
    foo(sp0 / 25);
    foo(sp0 / 26);
    temp_mach_4 = MULTU_HI(sp0, 0x2F684BDB);
    foo((u32) (temp_mach_4 + ((u32) (sp0 - temp_mach_4) >> 1)) >> 4);
    foo(sp0 / 28);
    foo(sp0 / 29);
    foo(sp0 / 30);
    temp_mach_5 = sp0 / 31;
    foo((u32) (temp_mach_5 + ((u32) (sp0 - temp_mach_5) >> 1)) >> 4);
    foo(u >> 5);
    foo(sp0 / 33);
    foo(sp0 / 100);
    foo(sp0 / 255);
    foo(sp0 / 360);
    foo(sp0 / 1000);
    foo(sp0 / 10000);
    foo(sp0 / 100000);
    foo(sp0 / 1000000);
    foo((u32) ((sp0 / 156250) * 4) >> 8);
    foo(sp0 / 100000000);
    foo((u32) ((u >> 0x10) * 4) >> 0x10);
    foo((u32) (((u32) MULTU_HI(sp0, 0xFFFFFFFD) >> 0x10) * 4) >> 0x10);
    foo((u32) (((u32) MULTU_HI((sp0 >> 1), 0x80000003) >> 0x10) * 8) >> 0x10);
    temp_mach_6 = sp0 / 1431655765;
    foo((u32) (((u32) (temp_mach_6 + ((u32) (sp0 - temp_mach_6) >> 1)) >> 0x10) * 4) >> 0x10);
    foo(u >> 0x1F);
    foo(sp0 >= 0x80000001U);
    foo(sp0 >= -2U);
    foo(sp0 >= -1U);
}

void test_u32_mod(u32 u) {
    u32 sp0;
    s32 temp_mach;
    s32 temp_mach_2;
    s32 temp_mach_3;
    s32 temp_mach_4;
    s32 temp_mach_5;
    s32 temp_mach_6;

    sp0 = u;
    foo(0U);
    foo(sp0 & 1);
    foo(u % 3);
    foo(sp0 & 3);
    foo(u % 5);
    foo(u % 6);
    temp_mach = u / 7;
    foo(u - (((u32) (temp_mach + ((u32) (u - temp_mach) >> 1)) >> 2) * 7));
    foo(sp0 & 7);
    foo(u % 9);
    foo(u % 10);
    foo(u % 11);
    foo(u % 12);
    foo(u % 13);
    foo(u % 14);
    foo(u % 15);
    foo(sp0 & 0xF);
    foo(u % 17);
    foo(u % 18);
    temp_mach_2 = MULTU_HI(u, 0xAF286BCB);
    foo(u - (((u32) (temp_mach_2 + ((u32) (u - temp_mach_2) >> 1)) >> 4) * 0x13));
    foo(u % 20);
    temp_mach_3 = MULTU_HI(u, 0x86186187);
    foo(u - (((u32) (temp_mach_3 + ((u32) (u - temp_mach_3) >> 1)) >> 4) * 0x15));
    foo(u % 22);
    foo(u % 23);
    foo(u % 24);
    foo(u % 25);
    foo(u % 26);
    temp_mach_4 = MULTU_HI(u, 0x2F684BDB);
    foo(u - (((u32) (temp_mach_4 + ((u32) (u - temp_mach_4) >> 1)) >> 4) * 0x1B));
    foo(u % 28);
    foo(u % 29);
    foo(u % 30);
    temp_mach_5 = u / 31;
    foo(u - (((u32) (temp_mach_5 + ((u32) (u - temp_mach_5) >> 1)) >> 4) * 0x1F));
    foo(sp0 & 0x1F);
    foo(u % 33);
    foo(u % 100);
    foo(u % 255);
    foo(u % 360);
    foo(u % 1000);
    foo(u % 10000);
    foo(u % 100000);
    foo(u % 1000000);
    foo(u - (((u32) ((u / 156250) * 4) >> 8) * 0x989680));
    foo(u % 100000000);
    foo(u & 0x3FFFFFFF);
    foo(u - (((u32) (((u32) MULTU_HI(u, 0xFFFFFFFD) >> 0x10) * 4) >> 0x10) * 0x40000001));
    foo(sp0 - (((u32) (((u32) MULTU_HI((sp0 >> 1), 0x80000003) >> 0x10) * 8) >> 0x10) * 0x7FFFFFFE));
    temp_mach_6 = sp0 / 1431655765;
    foo(sp0 - (((u32) (((u32) (temp_mach_6 + ((u32) (sp0 - temp_mach_6) >> 1)) >> 0x10) * 4) >> 0x10) * 0x7FFFFFFF));
    foo(sp0 & 0x7FFFFFFF);
    foo(sp0 - ((sp0 >= 0x80000001U) * 0x80000001));
    foo(sp0 + ((sp0 >= -2U) * 2));
    foo(sp0 + (sp0 >= -1U));
}
