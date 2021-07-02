s32 test(void) {
    static_int *= 0x1C8;
    extern_float = (f32) (extern_float * 456.0f);
    static_fn(&static_A);
    extern_fn(static_A_ptr);
    *static_bss_array = *static_array + *static_ro_array;
    return static_int;
}
