void test(struct A *a, s32 b) {
    D_410110 = (int *) a->array[b];
    D_410110 = (int *) &a->array[b];
    D_410110 = (int *) a->array2[b].x;
    D_410110 = &a->array2[b].x;
    D_410110 = (int *) a[b].y;
    D_410110 = (int *) a->array2[3].x;
    D_410110 = &a->array2[3].x;
}
