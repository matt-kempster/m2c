void func_00400098(s8 arg0);                        /* static */
void func_004003A8(s16 arg0);                       /* static */
void func_004007C0(u32 arg0);                       /* static */
void func_00400F48(u32 arg0);                       /* static */
void func_00401AA0(u32 arg0);                       /* static */
void func_00401FC4(u32 arg0);                       /* static */
extern ? D_8421085;
extern ? D_A7C5AC5;

void test(u32 a) {
    func_00400098((s8) a);
    func_004003A8((s16) a);
    func_004007C0(a);
    func_00400F48(a);
    func_00401AA0(a);
    func_00401FC4(a);
}

void func_00400098(s8 arg0) {
    func_00400090((u32) ((s32) ((s8) subroutine_arg4 + ((u32) (subroutine_arg4 << 0x18) >> 0x1F)) >> 1), arg0);
    func_00400090((u32) (s8) (subroutine_arg4 / 3));
    func_00400090((u32) (s8) (subroutine_arg4 / 5));
    func_00400090((u32) (s8) (subroutine_arg4 / 7));
    func_00400090((u32) (s8) (subroutine_arg4 / 0xA));
    func_00400090((u32) (s8) (subroutine_arg4 / 0x64));
    func_00400090(subroutine_arg4 / 0xFF);
    func_00400090((u32) (s8) ((s8) subroutine_arg4 - (((s32) ((s8) subroutine_arg4 + ((u32) (subroutine_arg4 << 0x18) >> 0x1F)) >> 1) * 2)));
    func_00400090((u32) (s8) (subroutine_arg4 % 3));
    func_00400090((u32) (s8) (subroutine_arg4 % 5));
    func_00400090((u32) (s8) (subroutine_arg4 % 7));
    func_00400090((u32) (s8) (subroutine_arg4 % 0xA));
    func_00400090((u32) (s8) (subroutine_arg4 % 0x64));
    func_00400090(subroutine_arg4 % 0xFF);
}

void func_004003A8(s16 arg0) {
    func_00400090((u32) ((s32) ((s16) subroutine_arg4 + ((u32) (subroutine_arg4 << 0x10) >> 0x1F)) >> 1), arg0);
    func_00400090((u32) (s16) (subroutine_arg4 / 3));
    func_00400090((u32) (s16) (subroutine_arg4 / 5));
    func_00400090((u32) (s16) (subroutine_arg4 / 7));
    func_00400090((u32) (s16) (subroutine_arg4 / 0xA));
    func_00400090((u32) (s16) (subroutine_arg4 / 0x64));
    func_00400090((u32) (s16) (subroutine_arg4 / 0xFF));
    func_00400090((u32) (s16) (subroutine_arg4 / 0x168));
    func_00400090(subroutine_arg4 / 0xFFFE);
    func_00400090((u32) (s16) ((s16) subroutine_arg4 - (((s32) ((s16) subroutine_arg4 + ((u32) (subroutine_arg4 << 0x10) >> 0x1F)) >> 1) * 2)));
    func_00400090((u32) (s16) (subroutine_arg4 % 3));
    func_00400090((u32) (s16) (subroutine_arg4 % 5));
    func_00400090((u32) (s16) (subroutine_arg4 % 7));
    func_00400090((u32) (s16) (subroutine_arg4 % 0xA));
    func_00400090((u32) (s16) (subroutine_arg4 % 0x64));
    func_00400090((u32) (s16) (subroutine_arg4 % 0xFF));
    func_00400090((u32) (s16) (subroutine_arg4 % 0x168));
    func_00400090(subroutine_arg4 % 0xFFFE);
}

void func_004007C0(u32 arg0) {
    s32 phi_a0;
    s32 phi_a0_2;
    s32 phi_a0_3;
    s32 phi_a0_4;
    s32 phi_a0_5;
    s32 phi_a0_6;

    func_00400090(arg0);
    func_00400090((u32) ((s32) (arg0 + (arg0 >> 0x1F)) >> 1));
    func_00400090(arg0 / 3);
    phi_a0 = (s32) arg0;
    if ((s32) arg0 < 0) {
        phi_a0 = arg0 + 3;
    }
    func_00400090((u32) (phi_a0 >> 2));
    func_00400090(arg0 / 5);
    func_00400090(arg0 / 6);
    func_00400090(arg0 / 7);
    phi_a0_2 = (s32) arg0;
    if ((s32) arg0 < 0) {
        phi_a0_2 = arg0 + 7;
    }
    func_00400090((u32) (phi_a0_2 >> 3));
    func_00400090(arg0 / 9);
    func_00400090(arg0 / 0xA);
    func_00400090(arg0 / 0xB);
    func_00400090(arg0 / 0xC);
    func_00400090(arg0 / 0xD);
    func_00400090(arg0 / 0xE);
    func_00400090(arg0 / 0xF);
    phi_a0_3 = (s32) arg0;
    if ((s32) arg0 < 0) {
        phi_a0_3 = arg0 + 0xF;
    }
    func_00400090((u32) (phi_a0_3 >> 4));
    func_00400090(arg0 / 0x11);
    func_00400090(arg0 / 0x12);
    func_00400090(arg0 / 0x13);
    func_00400090(arg0 / 0x14);
    func_00400090(arg0 / 0x15);
    func_00400090(arg0 / 0x16);
    func_00400090(arg0 / 0x17);
    func_00400090(arg0 / 0x18);
    func_00400090(arg0 / 0x19);
    func_00400090(arg0 / 0x1A);
    func_00400090(arg0 / 0x1B);
    func_00400090(arg0 / 0x1C);
    func_00400090(arg0 / 0x1D);
    func_00400090(arg0 / 0x1E);
    func_00400090(arg0 / 0x1F);
    phi_a0_4 = (s32) arg0;
    if ((s32) arg0 < 0) {
        phi_a0_4 = arg0 + 0x1F;
    }
    func_00400090((u32) (phi_a0_4 >> 5));
    func_00400090(arg0 / 0x21);
    func_00400090(arg0 / 0x64);
    func_00400090(arg0 / 0xFF);
    func_00400090(arg0 / 0x168);
    func_00400090(arg0 / 0x3E8);
    func_00400090(arg0 / 0x2710);
    func_00400090(arg0 / 0x186A0);
    func_00400090(arg0 / 0xF4240);
    func_00400090(arg0 / 0x989680);
    func_00400090(arg0 / 0x5F5E100);
    func_00400090(arg0 / 0x3FFFFFFE);
    func_00400090(arg0 / 0x3FFFFFFE);
    phi_a0_5 = (s32) arg0;
    if ((s32) arg0 < 0) {
        phi_a0_5 = arg0 + 0x3FFFFFFF;
    }
    func_00400090((u32) (phi_a0_5 >> 0x1E));
    func_00400090(arg0 / 0x40000000);
    func_00400090(arg0 / 0x80000000);
    func_00400090(arg0 / 0x7FFFFFFD);
    func_00400090(arg0 / 0x80000000);
    func_00400090(arg0 >> 0x1F);
    func_00400090(arg0 / -0x80000000);
    func_00400090(arg0 / -0x7FFFFFFD);
    func_00400090(arg0 / -0xA);
    func_00400090(arg0 / -7);
    func_00400090(arg0 / -5);
    phi_a0_6 = (s32) arg0;
    if ((s32) arg0 < 0) {
        phi_a0_6 = arg0 + 3;
    }
    func_00400090((u32) -(s32) (phi_a0_6 >> 2));
    func_00400090(arg0 / -3);
    func_00400090((u32) -(s32) ((s32) (arg0 + (arg0 >> 0x1F)) >> 1));
    func_00400090((u32) -(s32) arg0);
}

void func_00400F48(s32 arg0) {
    s32 phi_a0;
    s32 phi_a0_2;
    s32 phi_a0_3;
    s32 phi_a0_4;
    s32 phi_a0_5;
    s32 phi_a0_6;

    func_00400090(0U);
    func_00400090(arg0 % 2);
    func_00400090(arg0 % 3);
    phi_a0 = arg0;
    if (arg0 < 0) {
        phi_a0 = arg0 + 3;
    }
    func_00400090(arg0 - ((phi_a0 >> 2) * 4));
    func_00400090(arg0 % 5);
    func_00400090(arg0 % 6);
    func_00400090(arg0 % 7);
    phi_a0_2 = arg0;
    if (arg0 < 0) {
        phi_a0_2 = arg0 + 7;
    }
    func_00400090(arg0 - ((phi_a0_2 >> 3) * 8));
    func_00400090(arg0 % 9);
    func_00400090(arg0 % 0xA);
    func_00400090(arg0 % 0xB);
    func_00400090(arg0 % 0xC);
    func_00400090(arg0 % 0xD);
    func_00400090(arg0 % 0xE);
    func_00400090(arg0 % 0xF);
    phi_a0_3 = arg0;
    if (arg0 < 0) {
        phi_a0_3 = arg0 + 0xF;
    }
    func_00400090(arg0 - ((phi_a0_3 >> 4) * 0x10));
    func_00400090(arg0 % 0x11);
    func_00400090(arg0 % 0x12);
    func_00400090(arg0 % 0x13);
    func_00400090(arg0 % 0x14);
    func_00400090(arg0 % 0x15);
    func_00400090(arg0 % 0x16);
    func_00400090(arg0 % 0x17);
    func_00400090(arg0 % 0x18);
    func_00400090(arg0 % 0x19);
    func_00400090(arg0 % 0x1A);
    func_00400090(arg0 % 0x1B);
    func_00400090(arg0 % 0x1C);
    func_00400090(arg0 % 0x1D);
    func_00400090(arg0 % 0x1E);
    func_00400090(arg0 % 0x1F);
    phi_a0_4 = arg0;
    if (arg0 < 0) {
        phi_a0_4 = arg0 + 0x1F;
    }
    func_00400090(arg0 - ((phi_a0_4 >> 5) << 5));
    func_00400090(arg0 % 0x21);
    func_00400090(arg0 % 0x64);
    func_00400090(arg0 % 0xFF);
    func_00400090(arg0 % 0x168);
    func_00400090(arg0 % 0x3E8);
    func_00400090(arg0 % 0x2710);
    func_00400090(arg0 % 0x186A0);
    func_00400090(arg0 % 0xF4240);
    func_00400090(arg0 % 0x989680);
    func_00400090(arg0 % 0x5F5E100);
    func_00400090(arg0 % 0x3FFFFFFE);
    func_00400090(arg0 - ((arg0 / 0x3FFFFFFE) * 0x3FFFFFFF));
    phi_a0_5 = arg0;
    if (arg0 < 0) {
        phi_a0_5 = arg0 + 0x3FFFFFFF;
    }
    func_00400090(arg0 - ((phi_a0_5 >> 0x1E) << 0x1E));
    func_00400090(arg0 - ((arg0 / 0x40000000) * 0x40000001));
    func_00400090(arg0 - ((arg0 / 0x80000000) * 0x7FFFFFFD));
    func_00400090(arg0 - ((arg0 / 0x7FFFFFFD) * 0x7FFFFFFE));
    func_00400090(arg0 - ((arg0 / 0x80000000) * 0x7FFFFFFF));
    func_00400090(arg0 & 0x7FFFFFFF);
    func_00400090(arg0 - ((arg0 / 0x80000000) * 0x7FFFFFFF));
    func_00400090(arg0 - ((arg0 / 0x7FFFFFFD) * 0x7FFFFFFE));
    func_00400090(arg0 % 0xA);
    func_00400090(arg0 % 7);
    func_00400090(arg0 % 5);
    phi_a0_6 = arg0;
    if (arg0 < 0) {
        phi_a0_6 = arg0 + 3;
    }
    func_00400090(arg0 - ((phi_a0_6 >> 2) * 4));
    func_00400090(arg0 % 3);
    func_00400090(arg0 % 2);
    func_00400090(0U);
}

void func_00401AA0(u32 arg0) {
    s32 temp_hi;
    s32 temp_hi_2;
    s32 temp_hi_3;
    s32 temp_hi_4;
    s32 temp_hi_5;
    s32 temp_hi_6;

    func_00400090(arg0);
    func_00400090(arg0 >> 1);
    func_00400090(arg0 / 3);
    func_00400090(arg0 >> 2);
    func_00400090(arg0 / 5);
    func_00400090(arg0 / 6);
    temp_hi = arg0 / 7;
    func_00400090((u32) (temp_hi + ((u32) (arg0 - temp_hi) >> 1)) >> 2);
    func_00400090(arg0 >> 3);
    func_00400090(arg0 / 9);
    func_00400090(arg0 / 0xA);
    func_00400090(arg0 / 0xB);
    func_00400090(arg0 / 0xC);
    func_00400090(arg0 / 0xD);
    func_00400090(arg0 / 0xE);
    func_00400090(arg0 / 0xF);
    func_00400090(arg0 >> 4);
    func_00400090(arg0 / 0x11);
    func_00400090(arg0 / 0x12);
    temp_hi_2 = MULTU_HI(arg0, 0xAF286BCB);
    func_00400090((u32) (temp_hi_2 + ((u32) (arg0 - temp_hi_2) >> 1)) >> 4);
    func_00400090(arg0 / 0x14);
    temp_hi_3 = MULTU_HI(arg0, 0x86186187);
    func_00400090((u32) (temp_hi_3 + ((u32) (arg0 - temp_hi_3) >> 1)) >> 4);
    func_00400090(arg0 / 0x16);
    func_00400090(arg0 / 0x17);
    func_00400090(arg0 / 0x18);
    func_00400090(arg0 / 0x19);
    func_00400090(arg0 / 0x1A);
    temp_hi_4 = MULTU_HI(arg0, 0x2F684BDB);
    func_00400090((u32) (temp_hi_4 + ((u32) (arg0 - temp_hi_4) >> 1)) >> 4);
    func_00400090(arg0 / 0x1C);
    func_00400090(arg0 / 0x1D);
    func_00400090(arg0 / 0x1E);
    temp_hi_5 = MULTU_HI(arg0, ((s32) &D_8421085 | 0));
    func_00400090((u32) (temp_hi_5 + ((u32) (arg0 - temp_hi_5) >> 1)) >> 4);
    func_00400090(arg0 >> 5);
    func_00400090(arg0 / 0x21);
    func_00400090(arg0 / 0x64);
    func_00400090(arg0 / 0xFF);
    func_00400090(arg0 / 0x168);
    func_00400090(arg0 / 0x3E8);
    func_00400090(arg0 / 0x2710);
    func_00400090((u32) MULTU_HI((arg0 >> 5), ((s32) &D_A7C5AC5 | 0)) >> 7);
    func_00400090(arg0 / 0xF4240);
    func_00400090(arg0 / 0x989680);
    func_00400090(arg0 / 0x5F5E100);
    func_00400090(arg0 >> 0x1E);
    func_00400090((u32) MULTU_HI(arg0, -3) >> 0x1E);
    func_00400090(arg0 / 0x7FFFFFFD);
    temp_hi_6 = arg0 / 0x55555555;
    func_00400090((u32) (temp_hi_6 + ((u32) (arg0 - temp_hi_6) >> 1)) >> 0x1E);
    func_00400090(arg0 >> 0x1F);
    func_00400090((arg0 < 0x80000001U) ^ 1);
    func_00400090((arg0 < 0xFFFFFFFEU) ^ 1);
    func_00400090((arg0 < 0xFFFFFFFFU) ^ 1);
}

void func_00401FC4(u32 arg0) {
    s32 temp_hi;
    s32 temp_hi_2;
    s32 temp_hi_3;
    s32 temp_hi_4;
    s32 temp_hi_5;
    s32 temp_hi_6;

    func_00400090(arg0);
    func_00400090(arg0 >> 1);
    func_00400090(arg0 / 3);
    func_00400090(arg0 >> 2);
    func_00400090(arg0 / 5);
    func_00400090(arg0 / 6);
    temp_hi = arg0 / 7;
    func_00400090((u32) (temp_hi + ((u32) (arg0 - temp_hi) >> 1)) >> 2);
    func_00400090(arg0 >> 3);
    func_00400090(arg0 / 9);
    func_00400090(arg0 / 0xA);
    func_00400090(arg0 / 0xB);
    func_00400090(arg0 / 0xC);
    func_00400090(arg0 / 0xD);
    func_00400090(arg0 / 0xE);
    func_00400090(arg0 / 0xF);
    func_00400090(arg0 >> 4);
    func_00400090(arg0 / 0x11);
    func_00400090(arg0 / 0x12);
    temp_hi_2 = MULTU_HI(arg0, 0xAF286BCB);
    func_00400090((u32) (temp_hi_2 + ((u32) (arg0 - temp_hi_2) >> 1)) >> 4);
    func_00400090(arg0 / 0x14);
    temp_hi_3 = MULTU_HI(arg0, 0x86186187);
    func_00400090((u32) (temp_hi_3 + ((u32) (arg0 - temp_hi_3) >> 1)) >> 4);
    func_00400090(arg0 / 0x16);
    func_00400090(arg0 / 0x17);
    func_00400090(arg0 / 0x18);
    func_00400090(arg0 / 0x19);
    func_00400090(arg0 / 0x1A);
    temp_hi_4 = MULTU_HI(arg0, 0x2F684BDB);
    func_00400090((u32) (temp_hi_4 + ((u32) (arg0 - temp_hi_4) >> 1)) >> 4);
    func_00400090(arg0 / 0x1C);
    func_00400090(arg0 / 0x1D);
    func_00400090(arg0 / 0x1E);
    temp_hi_5 = MULTU_HI(arg0, ((s32) &D_8421085 | 0));
    func_00400090((u32) (temp_hi_5 + ((u32) (arg0 - temp_hi_5) >> 1)) >> 4);
    func_00400090(arg0 >> 5);
    func_00400090(arg0 / 0x21);
    func_00400090(arg0 / 0x64);
    func_00400090(arg0 / 0xFF);
    func_00400090(arg0 / 0x168);
    func_00400090(arg0 / 0x3E8);
    func_00400090(arg0 / 0x2710);
    func_00400090((u32) MULTU_HI((arg0 >> 5), ((s32) &D_A7C5AC5 | 0)) >> 7);
    func_00400090(arg0 / 0xF4240);
    func_00400090(arg0 / 0x989680);
    func_00400090(arg0 / 0x5F5E100);
    func_00400090(arg0 >> 0x1E);
    func_00400090((u32) MULTU_HI(arg0, -3) >> 0x1E);
    func_00400090(arg0 / 0x7FFFFFFD);
    temp_hi_6 = arg0 / 0x55555555;
    func_00400090((u32) (temp_hi_6 + ((u32) (arg0 - temp_hi_6) >> 1)) >> 0x1E);
    func_00400090(arg0 >> 0x1F);
    func_00400090((arg0 < 0x80000001U) ^ 1);
    func_00400090((arg0 < 0xFFFFFFFEU) ^ 1);
    func_00400090((arg0 < 0xFFFFFFFFU) ^ 1);
}
