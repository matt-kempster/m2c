extern f32 g;

u8 test(f32 arg0) {
    f32 temp_fcmp_rhs;

    temp_fcmp_rhs = g;
    g = 0.0f;
    return (u8) (arg0 > temp_fcmp_rhs);
}
