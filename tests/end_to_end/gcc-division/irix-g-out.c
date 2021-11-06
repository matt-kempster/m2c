void func_004000A4(u32 arg0);                       /* static */
void func_00400224(u32 arg0);                       /* static */
void func_00400404(u32 arg0);                       /* static */
void func_004009F4(u32 arg0);                       /* static */
void func_00400FFC(u32 arg0);                       /* static */
void func_00401498(u32 arg0);                       /* static */

void test(u32 a) {
    func_004000A4(a);
    func_00400224(a);
    func_00400404(a);
    func_004009F4(a);
    func_00400FFC(a);
    func_00401498(a);
}

void func_004000A4(s8 arg0) {
    func_00400090((u32) ((s32) arg0 / 2));
    func_00400090((u32) ((s32) arg0 / 3));
    func_00400090((u32) ((s32) arg0 / 5));
    func_00400090((u32) ((s32) arg0 / 7));
    func_00400090((u32) ((s32) arg0 / 0xA));
    func_00400090((u32) ((s32) arg0 / 0x64));
    func_00400090((u32) ((s32) arg0 / 0xFF));
    func_00400090((u32) ((s32) arg0 % 2));
    func_00400090((u32) ((s32) arg0 % 3));
    func_00400090((u32) ((s32) arg0 % 5));
    func_00400090((u32) ((s32) arg0 % 7));
    func_00400090((u32) ((s32) arg0 % 0xA));
    func_00400090((u32) ((s32) arg0 % 0x64));
    func_00400090((u32) ((s32) arg0 % 0xFF));
}

void func_00400224(s16 arg0) {
    func_00400090((u32) ((s32) arg0 / 2));
    func_00400090((u32) ((s32) arg0 / 3));
    func_00400090((u32) ((s32) arg0 / 5));
    func_00400090((u32) ((s32) arg0 / 7));
    func_00400090((u32) ((s32) arg0 / 0xA));
    func_00400090((u32) ((s32) arg0 / 0x64));
    func_00400090((u32) ((s32) arg0 / 0xFF));
    func_00400090((u32) ((s32) arg0 / 0x168));
    func_00400090((u32) ((s32) arg0 / 0xFFFE));
    func_00400090((u32) ((s32) arg0 % 2));
    func_00400090((u32) ((s32) arg0 % 3));
    func_00400090((u32) ((s32) arg0 % 5));
    func_00400090((u32) ((s32) arg0 % 7));
    func_00400090((u32) ((s32) arg0 % 0xA));
    func_00400090((u32) ((s32) arg0 % 0x64));
    func_00400090((u32) ((s32) arg0 % 0xFF));
    func_00400090((u32) ((s32) arg0 % 0x168));
    func_00400090((u32) ((s32) arg0 % 0xFFFE));
}

void func_00400404(u32 arg0) {
    s32 phi_at;

    func_00400090(arg0);
    func_00400090((u32) ((s32) arg0 / 2));
    func_00400090((u32) ((s32) arg0 / 3));
    func_00400090((u32) ((s32) arg0 / 4));
    func_00400090((u32) ((s32) arg0 / 5));
    func_00400090((u32) ((s32) arg0 / 6));
    func_00400090((u32) ((s32) arg0 / 7));
    func_00400090((u32) ((s32) arg0 / 8));
    func_00400090((u32) ((s32) arg0 / 9));
    func_00400090((u32) ((s32) arg0 / 0xA));
    func_00400090((u32) ((s32) arg0 / 0xB));
    func_00400090((u32) ((s32) arg0 / 0xC));
    func_00400090((u32) ((s32) arg0 / 0xD));
    func_00400090((u32) ((s32) arg0 / 0xE));
    func_00400090((u32) ((s32) arg0 / 0xF));
    func_00400090((u32) ((s32) arg0 / 0x10));
    func_00400090((u32) ((s32) arg0 / 0x11));
    func_00400090((u32) ((s32) arg0 / 0x12));
    func_00400090((u32) ((s32) arg0 / 0x13));
    func_00400090((u32) ((s32) arg0 / 0x14));
    func_00400090((u32) ((s32) arg0 / 0x15));
    func_00400090((u32) ((s32) arg0 / 0x16));
    func_00400090((u32) ((s32) arg0 / 0x17));
    func_00400090((u32) ((s32) arg0 / 0x18));
    func_00400090((u32) ((s32) arg0 / 0x19));
    func_00400090((u32) ((s32) arg0 / 0x1A));
    func_00400090((u32) ((s32) arg0 / 0x1B));
    func_00400090((u32) ((s32) arg0 / 0x1C));
    func_00400090((u32) ((s32) arg0 / 0x1D));
    func_00400090((u32) ((s32) arg0 / 0x1E));
    func_00400090((u32) ((s32) arg0 / 0x1F));
    func_00400090((u32) ((s32) arg0 / 0x20));
    func_00400090((u32) ((s32) arg0 / 0x21));
    func_00400090((u32) ((s32) arg0 / 0x64));
    func_00400090((u32) ((s32) arg0 / 0xFF));
    func_00400090((u32) ((s32) arg0 / 0x168));
    func_00400090((u32) ((s32) arg0 / 0x3E8));
    func_00400090((u32) ((s32) arg0 / 0x2710));
    func_00400090((u32) ((s32) arg0 / 0x186A0));
    func_00400090((u32) ((s32) arg0 / 0xF4240));
    func_00400090((u32) ((s32) arg0 / 0x979680));
    func_00400090((u32) ((s32) arg0 / 0x54FE100));
    func_00400090((u32) ((s32) arg0 / 0x3FFFFFFE));
    func_00400090((u32) ((s32) arg0 / 0x3FFFFFFF));
    phi_at = (s32) arg0;
    if ((s32) arg0 < 0) {
        phi_at = arg0 + 0x3FFFFFFF;
    }
    func_00400090((u32) (phi_at >> 0x1E));
    func_00400090((u32) ((s32) arg0 / 0x40000001));
    func_00400090((u32) ((s32) arg0 / 0x7FFFFFFD));
    func_00400090((u32) ((s32) arg0 / 0x7FFFFFFE));
    func_00400090((u32) ((s32) arg0 / 0x7FFFFFFF));
    func_00400090(arg0 / 0x80000000U);
    func_00400090((u32) ((s32) arg0 / 0x80000001));
    func_00400090((u32) ((s32) arg0 / 0x80000002));
    func_00400090((u32) ((s32) arg0 / -0xA));
    func_00400090((u32) ((s32) arg0 / -7));
    func_00400090((u32) ((s32) arg0 / -5));
    func_00400090((u32) -(s32) ((s32) arg0 / 4));
    func_00400090((u32) ((s32) arg0 / -3));
    func_00400090((u32) -(s32) ((s32) arg0 / 2));
    func_00400090((u32) -(s32) arg0);
}

void func_004009F4(s32 arg0) {
    u32 temp_a0;
    u32 phi_a0;

    func_00400090(0U);
    func_00400090((u32) (arg0 % 2));
    func_00400090((u32) (arg0 % 3));
    func_00400090((u32) (arg0 % 4));
    func_00400090((u32) (arg0 % 5));
    func_00400090((u32) (arg0 % 6));
    func_00400090((u32) (arg0 % 7));
    func_00400090((u32) (arg0 % 8));
    func_00400090((u32) (arg0 % 9));
    func_00400090((u32) (arg0 % 0xA));
    func_00400090((u32) (arg0 % 0xB));
    func_00400090((u32) (arg0 % 0xC));
    func_00400090((u32) (arg0 % 0xD));
    func_00400090((u32) (arg0 % 0xE));
    func_00400090((u32) (arg0 % 0xF));
    func_00400090((u32) (arg0 % 0x10));
    func_00400090((u32) (arg0 % 0x11));
    func_00400090((u32) (arg0 % 0x12));
    func_00400090((u32) (arg0 % 0x13));
    func_00400090((u32) (arg0 % 0x14));
    func_00400090((u32) (arg0 % 0x15));
    func_00400090((u32) (arg0 % 0x16));
    func_00400090((u32) (arg0 % 0x17));
    func_00400090((u32) (arg0 % 0x18));
    func_00400090((u32) (arg0 % 0x19));
    func_00400090((u32) (arg0 % 0x1A));
    func_00400090((u32) (arg0 % 0x1B));
    func_00400090((u32) (arg0 % 0x1C));
    func_00400090((u32) (arg0 % 0x1D));
    func_00400090((u32) (arg0 % 0x1E));
    func_00400090((u32) (arg0 % 0x1F));
    func_00400090((u32) (arg0 % 0x20));
    func_00400090((u32) (arg0 % 0x21));
    func_00400090((u32) (arg0 % 0x64));
    func_00400090((u32) (arg0 % 0xFF));
    func_00400090((u32) (arg0 % 0x168));
    func_00400090((u32) (arg0 % 0x3E8));
    func_00400090((u32) (arg0 % 0x2710));
    func_00400090((u32) (arg0 % 0x186A0));
    func_00400090((u32) (arg0 % 0xF4240));
    func_00400090((u32) (arg0 % 0x979680));
    func_00400090((u32) (arg0 % 0x54FE100));
    func_00400090((u32) (arg0 % 0x3FFFFFFE));
    func_00400090((u32) (arg0 % 0x3FFFFFFF));
    temp_a0 = arg0 & 0x3FFFFFFF;
    phi_a0 = temp_a0;
    if ((arg0 < 0) && (temp_a0 != 0)) {
        phi_a0 = temp_a0 - 0x40000000;
    }
    func_00400090(phi_a0);
    func_00400090((u32) (arg0 % 0x40000001));
    func_00400090((u32) (arg0 % 0x7FFFFFFD));
    func_00400090((u32) (arg0 % 0x7FFFFFFE));
    func_00400090((u32) (arg0 % 0x7FFFFFFF));
    func_00400090((u32) arg0 % 0x80000000U);
    func_00400090((u32) (arg0 % 0x80000001));
    func_00400090((u32) (arg0 % 0x80000002));
    func_00400090((u32) (arg0 % -0xA));
    func_00400090((u32) (arg0 % -7));
    func_00400090((u32) (arg0 % -5));
    func_00400090((u32) (arg0 % 4));
    func_00400090((u32) (arg0 % -3));
    func_00400090((u32) (arg0 % 2));
    func_00400090(0U);
}

void func_00400FFC(u32 arg0) {
    func_00400090(arg0);
    func_00400090(arg0 >> 1);
    func_00400090(arg0 / 3U);
    func_00400090(arg0 >> 2);
    func_00400090(arg0 / 5U);
    func_00400090(arg0 / 6U);
    func_00400090(arg0 / 7U);
    func_00400090(arg0 >> 3);
    func_00400090(arg0 / 9U);
    func_00400090(arg0 / 0xAU);
    func_00400090(arg0 / 0xBU);
    func_00400090(arg0 / 0xCU);
    func_00400090(arg0 / 0xDU);
    func_00400090(arg0 / 0xEU);
    func_00400090(arg0 / 0xFU);
    func_00400090(arg0 >> 4);
    func_00400090(arg0 / 0x11U);
    func_00400090(arg0 / 0x12U);
    func_00400090(arg0 / 0x13U);
    func_00400090(arg0 / 0x14U);
    func_00400090(arg0 / 0x15U);
    func_00400090(arg0 / 0x16U);
    func_00400090(arg0 / 0x17U);
    func_00400090(arg0 / 0x18U);
    func_00400090(arg0 / 0x19U);
    func_00400090(arg0 / 0x1AU);
    func_00400090(arg0 / 0x1BU);
    func_00400090(arg0 / 0x1CU);
    func_00400090(arg0 / 0x1DU);
    func_00400090(arg0 / 0x1EU);
    func_00400090(arg0 / 0x1FU);
    func_00400090(arg0 >> 5);
    func_00400090(arg0 / 0x21U);
    func_00400090(arg0 / 0x64U);
    func_00400090(arg0 / 0xFFU);
    func_00400090(arg0 / 0x168U);
    func_00400090(arg0 / 0x3E8U);
    func_00400090(arg0 / 0x2710U);
    func_00400090(arg0 / 0x186A0U);
    func_00400090(arg0 / 0xF4240U);
    func_00400090(arg0 / 0x979680U);
    func_00400090(arg0 / 0x54FE100U);
    func_00400090(arg0 >> 0x1E);
    func_00400090(arg0 / 0x40000001U);
    func_00400090(arg0 / 0x7FFFFFFEU);
    func_00400090(arg0 / 0x7FFFFFFFU);
    func_00400090(arg0 / 0x80000000U);
    func_00400090(arg0 / 0x80000001U);
    func_00400090(arg0 / -2U);
    func_00400090(arg0 / -1U);
}

void func_00401498(u32 arg0) {
    func_00400090(arg0);
    func_00400090(arg0 >> 1);
    func_00400090(arg0 / 3U);
    func_00400090(arg0 >> 2);
    func_00400090(arg0 / 5U);
    func_00400090(arg0 / 6U);
    func_00400090(arg0 / 7U);
    func_00400090(arg0 >> 3);
    func_00400090(arg0 / 9U);
    func_00400090(arg0 / 0xAU);
    func_00400090(arg0 / 0xBU);
    func_00400090(arg0 / 0xCU);
    func_00400090(arg0 / 0xDU);
    func_00400090(arg0 / 0xEU);
    func_00400090(arg0 / 0xFU);
    func_00400090(arg0 >> 4);
    func_00400090(arg0 / 0x11U);
    func_00400090(arg0 / 0x12U);
    func_00400090(arg0 / 0x13U);
    func_00400090(arg0 / 0x14U);
    func_00400090(arg0 / 0x15U);
    func_00400090(arg0 / 0x16U);
    func_00400090(arg0 / 0x17U);
    func_00400090(arg0 / 0x18U);
    func_00400090(arg0 / 0x19U);
    func_00400090(arg0 / 0x1AU);
    func_00400090(arg0 / 0x1BU);
    func_00400090(arg0 / 0x1CU);
    func_00400090(arg0 / 0x1DU);
    func_00400090(arg0 / 0x1EU);
    func_00400090(arg0 / 0x1FU);
    func_00400090(arg0 >> 5);
    func_00400090(arg0 / 0x21U);
    func_00400090(arg0 / 0x64U);
    func_00400090(arg0 / 0xFFU);
    func_00400090(arg0 / 0x168U);
    func_00400090(arg0 / 0x3E8U);
    func_00400090(arg0 / 0x2710U);
    func_00400090(arg0 / 0x186A0U);
    func_00400090(arg0 / 0xF4240U);
    func_00400090(arg0 / 0x979680U);
    func_00400090(arg0 / 0x54FE100U);
    func_00400090(arg0 >> 0x1E);
    func_00400090(arg0 / 0x40000001U);
    func_00400090(arg0 / 0x7FFFFFFEU);
    func_00400090(arg0 / 0x7FFFFFFFU);
    func_00400090(arg0 / 0x80000000U);
    func_00400090(arg0 / 0x80000001U);
    func_00400090(arg0 / -2U);
    func_00400090(arg0 / -1U);
}
