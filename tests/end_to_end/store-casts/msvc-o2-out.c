void test(struct A *a, s32 b, s16 c) {
    a->y = (s16) b;
    a->x = 2;
    a->z = (s16) (s32) c;
    a->w = b + (s32) c;
}
