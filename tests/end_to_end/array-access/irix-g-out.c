void test(struct A *a, s32 b) {
    D_410130 = (int *) a->array[b];
    D_410130 = &a->array[b];
    D_410130 = (s32) a->array2[b].x;
    D_410130 = &a->array2[b].x;
    D_410130 = (s32) a[b].y;
}
