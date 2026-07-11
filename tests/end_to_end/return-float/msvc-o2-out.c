static f32 real_41700000 = 15.0f;                   /* const */
static f32 real_00000000 = 0.0f;                    /* const */

f32 test(f32 x) {
    if (x != real_00000000) {
        return real_41700000;
    }
    return x;
}
