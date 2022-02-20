s32 func_00400090(s8 *, ?);                         /* static */
? func_004000A0(s32, s8 *, s32);                    /* static */

void test(s32 arg1, s8 arg8, ? arg48000) {
    s32 temp_a2;
    s32 temp_v0;
    s8 *temp_a1;
    s8 *temp_v0_2;

    arg48000.unk347C = arg1;
    temp_v0 = func_00400090(&arg8, 0x123456);
    temp_a1 = &arg8;
    temp_a2 = temp_v0;
    if (temp_v0 < 0) {
        return;
    }
    temp_v0_2 = &temp_a1[temp_a2];
    arg8 ^= 0x55;
    temp_v0_2->unk-1 = (s8) (temp_v0_2->unk-1 ^ 0x55);
    func_004000A0(arg48000.unk347C, temp_a1, temp_a2);
}
