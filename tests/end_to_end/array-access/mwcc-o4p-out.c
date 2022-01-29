void test(struct A *a, s32 b) {
    *NULL = a->array[b];
    *NULL = (s32 *) (a + ((b * 4) + 4));
    *NULL = (s32 *) a->array2[b].x;
    *NULL = (s32 *) (a + ((b * 8) + 0x30));
    *NULL = (s32 *) a[b].y;
    *NULL = (s32 *) a->array2[3].x;
    *NULL = &a->array2[3].x;
}
