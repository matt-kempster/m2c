f32 test(f32 number) {
    f32 sp0;                                        /* compiler-managed */

    sp0 = number;
    sp0 = 0x5F3759DF - ((bitwise s32) number >> 1);
    return (1.5f - (number * 0.5f * (bitwise f32) sp0 * (bitwise f32) sp0)) * (bitwise f32) sp0;
}
