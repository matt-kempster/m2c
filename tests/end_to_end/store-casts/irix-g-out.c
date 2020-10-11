void test(struct A *a, s32 b, s16 c) {
    s16 temp_a2;

    temp_a2 = c;
    a->x = 2;
    a->y = (s16) b;
    a->z = temp_a2;
    a->w = temp_a2 + b;
}
