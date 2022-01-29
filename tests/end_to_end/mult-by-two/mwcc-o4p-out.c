s32 test(s32 arg0) {
    *NULL = (f64) (*NULL * *NULL);
    *NULL = (f32) (*MIPS2C_ERROR(Read from unset register $r0) * *MIPS2C_ERROR(Read from unset register $r0));
    return arg0;
}
