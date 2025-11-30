struct _m2c_stack_test {
    /* 0x0 */ char pad0[0xC];
};                                                  /* size = 0xC */

s32 __mulsf3(s32, s32);                             /* extern */
M2C_UNK extern_fn(s32);                             /* extern */
M2C_UNK static_fn(s32);                             /* static */

s32 test(void) {
    s32 *temp_r4;

    static_int *= 0x1C8;
    temp_r4 = M2C_FIELD(&.L4, s32 **, 4);
    *temp_r4 = __mulsf3(*temp_r4, M2C_FIELD(&.L4, s32 *, 8));
    static_fn(M2C_FIELD(&.L4, s32 *, 0xC));
    extern_fn(*M2C_FIELD(&.L4, s32 **, 0x10));
    *M2C_FIELD(&.L4, s32 **, 0x14) = *M2C_FIELD(&.L4, s32 **, 0x18) + *M2C_FIELD(&.L4, s32 **, 0x1C);
    return static_int;
}
