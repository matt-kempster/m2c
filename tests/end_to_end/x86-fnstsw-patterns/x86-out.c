? _clobber_fsw();                                   /* extern */
extern f32 f32_a;
extern f32 f32_b;
extern f32 f32_c;
extern f64 f64_a;
extern f64 f64_b;
extern s16 i16_a;
extern s32 i32_a;
extern f32 lhs;
extern f32 rhs;

u8 test(void) {
    return (u8) (f32_a > f32_b);
}

u8 fcom_q(void) {
    return (u8) (f64_a > f64_b);
}

u8 fcom_reg(void) {
    return (u8) (f32_b > f32_a);
}

u8 fcomp_dword(void) {
    return (u8) (f32_a > f32_b);
}

u8 fucom_reg(void) {
    return (u8) (f32_b > f32_a);
}

u8 fucomp_q(void) {
    return (u8) (f64_a > f64_b);
}

u8 fcompp_case(void) {
    return (u8) (f32_b > f32_a);
}

u8 fucompp_case(void) {
    return (u8) (f32_b > f32_a);
}

u8 ftst_case(void) {
    return (u8) (f32_a > 0.0f);
}

u8 ficom_word(void) {
    return (u8) (f32_a > (f32) i16_a);
}

u8 ficom_dword(void) {
    return (u8) (f32_a > (f32) i32_a);
}

u8 ficomp_word(void) {
    return (u8) (f32_a > (f32) i16_a);
}

u8 ficomp_dword(void) {
    return (u8) (f32_a > (f32) i32_a);
}

u8 serial_last_wins(void) {
    return (u8) (f32_a > f32_c);
}

u8 cross_block(void) {
    return (u8) (f32_a > f32_b);
}

u8 phi_status(s32 ecx) {
    u16 var_fsw;

    if (ecx != 0) {
        var_fsw = M2C_FCMP(f32_a, f32_b);
    } else {
        var_fsw = M2C_FCMP(f32_a, f32_c);
    }
    return (u8) (((u8) (var_fsw >> 8) & 0x41) == 0);
}

s32 extra_status_use(void) {
    return (u8) (f32_a > f32_b) + M2C_FCMP(f32_a, f32_b);
}

u8 requires_compare(void) {
    return (u8) (((u8) (M2C_FNSTSW() >> 8) & 0x41) == 0);
}

u8 clobber_eax(void) {
    return (u8) (((u8) (0U >> 8) & 0x41) == 0);
}

u8 clobber_fsw(void) {
    _clobber_fsw();
    return (u8) (((u8) (M2C_FNSTSW() >> 8) & 0x41) == 0);
}

u8 unsupported_mask(void) {
    return (u8) (((u8) (M2C_FCMP(f32_a, f32_b) >> 8) & ~0x7F) == 0);
}

u8 interleaved(s32 edx) {
    return (u8) (f32_a > f32_b);
}

s32 unrelated_byte_test(s8 ecx) {
    return (u8) (f32_a > f32_b) + (u8) ((ecx & 0x41) != 0);
}

u8 capture_order(void) {
    f32 temp_f0;
    f32 temp_fcmp_rhs;

    temp_f0 = lhs;
    temp_fcmp_rhs = rhs;
    lhs = 0.0f;
    rhs = 0.0f;
    return (u8) (temp_f0 > temp_fcmp_rhs);
}

s32 unreachable_raw(void) {
    return 0;
}
