static f32 _real_41700000 = 15.0f;                  /* const */
static f32 _real_00000000 = 0.0f;                   /* const */

f32 test(f32 x) {
    if (x != _real_00000000) {
        return _real_41700000;
    }
    return x;
}
