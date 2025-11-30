struct _m2c_stack_test {
    /* 0x0 */ char pad0[0xC];
};                                                  /* size = 0xC */

s32 __mulsf3(s32, s32);                             /* extern */
M2C_UNK extern_fn(struct A *);                      /* extern */
M2C_UNK static_fn(struct A *);                      /* static */
extern s32 extern_float;

s32 test(void) {
    static_int *= 0x1C8;
    extern_float = __mulsf3(extern_float, 0x43E40000);
    static_fn(&static_A);
    extern_fn(static_A_ptr);
    *static_bss_array = *static_array + *static_ro_array;
    return static_int;
}
