MIPS2C_UNK extern_fn(struct A *); // extern
MIPS2C_UNK static_fn(struct A *); // static
extern f32 extern_float;
struct A *static_A_ptr = &static_A;
s32 static_int = 0;
struct A static_A; // type too large by 15; const

s32 test(void) {
    static_int *= 0x1C8;
    extern_float = (f32) (extern_float * 456.0f);
    static_fn(&static_A);
    extern_fn(static_A_ptr);
    return static_int;
}
