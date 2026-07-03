static f32 _real_3fc00000 = 1.5f;                   /* const */
static f32 _real_3f000000 = 0.5f;                   /* const */

f32 test(f32 arg0) {
    f32 sp0;                                        /* compiler-managed */

    sp0 = arg0;
    sp0 = 0x5F3759DF - ((bitwise s32) arg0 >> 1);
    return (_real_3fc00000 - (arg0 * _real_3f000000 * (bitwise f32) sp0 * (bitwise f32) sp0)) * (bitwise f32) sp0;
}
