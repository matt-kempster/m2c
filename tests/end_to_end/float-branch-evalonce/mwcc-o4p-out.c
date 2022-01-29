s32 test(s32 arg0) {
    *NULL = *MIPS2C_ERROR(Read from unset register $r0);
    if (*MIPS2C_ERROR(Read from unset register $r0) < *MIPS2C_ERROR(Read from unset register $r0)) {
        *NULL = (f32) *MIPS2C_ERROR(Read from unset register $r0);
    }
    *NULL = (f32) *MIPS2C_ERROR(Read from unset register $r0);
    if (!(*MIPS2C_ERROR(Read from unset register $r0) < *MIPS2C_ERROR(Read from unset register $r0))) {
        *NULL = (f32) *MIPS2C_ERROR(Read from unset register $r0);
        return arg0;
    }
    return arg0;
}
