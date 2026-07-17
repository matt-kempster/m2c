extern f32 g;

u8 test(f32 arg0) {
    f32 temp_marker_arg;

    temp_marker_arg = g;
    g = 0.0f;
    return (u8) (arg0 > temp_marker_arg);
}
