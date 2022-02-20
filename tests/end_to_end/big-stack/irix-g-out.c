s32 func_00400090(s32, s8 *, s32);                  /* static */
s32 func_004000B0(s32, s8 *, s32);                  /* static */

s32 test(s32 arg0, s32 arg1, s32 arg7, s8 arg8, ? arg48000) {
    s8 *temp_t3;

    arg48000.unk3478 = arg0;
    arg48000.unk347C = arg1;
    arg7 = 0x123456;
    arg7 = func_00400090(arg48000.unk3478, &arg8, arg7);
    if (arg7 < 0) {
        return arg7;
    }
    arg8 ^= 0x55;
    temp_t3 = &(&arg8)[arg7];
    temp_t3->unk-1 = (s8) (temp_t3->unk-1 ^ 0x55);
    return func_004000B0(arg48000.unk347C, &arg8, arg7);
}
