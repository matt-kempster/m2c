static f32 real_3fc00000 = 1.5f;                    /* const */
static f32 real_3f000000 = 0.5f;                    /* const */

f32 test(f32 number) {
    f32 sp0;                                        /* compiler-managed */

    sp0 = number;
    sp0 = 0x5F3759DF - ((bitwise s32) number >> 1);
    return (real_3fc00000 - (number * real_3f000000 * (bitwise f32) sp0 * (bitwise f32) sp0)) * (bitwise f32) sp0;
}
