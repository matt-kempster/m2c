void test(struct A *a, s32 b, s16 c) {
    u16 temp_r2;

    temp_r2 = (u16) c;
    a->x = 2;
    a->y = (s16) b;
    a->z = (s16) temp_r2;
    a->w = (s16) temp_r2 + b;
}
