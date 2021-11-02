void test(s32 x, u32 y) {
    s32 temp_hi;
    s32 temp_hi_2;
    s32 phi_a0;
    s32 phi_a0_2;
    s32 phi_a0_3;
    s32 phi_a0_4;
    s32 phi_a0_5;
    s32 phi_a0_6;

    func_00400090((u32) ((s32) (x + ((u32) x >> 0x1F)) >> 1));
    func_00400090(x / 3);
    phi_a0 = x;
    if (x < 0) {
        phi_a0 = x + 3;
    }
    func_00400090((u32) (phi_a0 >> 2));
    func_00400090(x / 5);
    func_00400090(x / 6);
    func_00400090(x / 7);
    phi_a0_2 = x;
    if (x < 0) {
        phi_a0_2 = x + 7;
    }
    func_00400090((u32) (phi_a0_2 >> 3));
    func_00400090(x / 9);
    func_00400090(x / 0xA);
    func_00400090(x / 0x64);
    func_00400090(x / 0xFF);
    phi_a0_3 = x;
    if (x < 0) {
        phi_a0_3 = x + 0xFF;
    }
    func_00400090((u32) (phi_a0_3 >> 8));
    func_00400090(x / 0x168);
    func_00400090(x / 0x3E8);
    func_00400090(x % 2);
    func_00400090(x % 3);
    phi_a0_4 = x;
    if (x < 0) {
        phi_a0_4 = x + 3;
    }
    func_00400090(x - ((phi_a0_4 >> 2) * 4));
    func_00400090(x % 5);
    func_00400090(x % 6);
    func_00400090(x % 7);
    phi_a0_5 = x;
    if (x < 0) {
        phi_a0_5 = x + 7;
    }
    func_00400090(x - ((phi_a0_5 >> 3) * 8));
    func_00400090(x % 9);
    func_00400090(x % 0xA);
    func_00400090(x % 0x64);
    func_00400090(x % 0xFF);
    phi_a0_6 = x;
    if (x < 0) {
        phi_a0_6 = x + 0xFF;
    }
    func_00400090(x - ((phi_a0_6 >> 8) << 8));
    func_00400090(x % 0x168);
    func_00400090(x % 0x3E8);
    func_00400090(y >> 1);
    func_00400090(y / 3);
    func_00400090(y >> 2);
    func_00400090(y / 5);
    func_00400090(y / 6);
    temp_hi = y / 7;
    func_00400090((u32) (temp_hi + ((u32) (y - temp_hi) >> 1)) >> 2);
    func_00400090(y >> 3);
    func_00400090(y / 9);
    func_00400090(y / 0xA);
    func_00400090(y / 0x64);
    func_00400090(y / 0xFF);
    func_00400090(y >> 8);
    func_00400090(y / 0x168);
    func_00400090(y / 0x3E8);
    func_00400090(y & 1);
    func_00400090(y % 3);
    func_00400090(y & 3);
    func_00400090(y % 5);
    func_00400090(y % 6);
    temp_hi_2 = y / 7;
    func_00400090(y - (((u32) (temp_hi_2 + ((u32) (y - temp_hi_2) >> 1)) >> 2) * 7));
    func_00400090(y & 7);
    func_00400090(y % 9);
    func_00400090(y % 0xA);
    func_00400090(y % 0x64);
    func_00400090(y % 0xFF);
    func_00400090(y & 0xFF);
    func_00400090(y % 0x168);
    func_00400090(y % 0x3E8);
}
