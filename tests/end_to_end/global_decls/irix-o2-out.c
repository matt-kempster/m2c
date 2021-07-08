MIPS2C_UNK extern_fn(struct A *); // extern
MIPS2C_UNK static_fn(struct A *); // static
extern f32 extern_float;
struct A static_A; // unable to generate initializer
struct A *static_A_ptr = &static_A;
s32 static_array[3] = {2, 4, 6};
struct A static_bss_A;
s32 static_bss_array[3];
s32 static_int;
s32 static_ro_array[3] = {7, 8, 9}; // const

s32 test(void) {
    static_int *= 0x1C8;
    extern_float = (f32) (extern_float * 456.0f);
    static_fn(&static_A);
    extern_fn(static_A_ptr);
    *static_bss_array = *static_array + *static_ro_array;
    return static_int;
}
