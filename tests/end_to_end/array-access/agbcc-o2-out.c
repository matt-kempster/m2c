void test(struct A *a, s32 b) {
    glob = (s32) a->array[b];
    glob = (s32) &a->array[b];
    glob = *(&a->array2[0].x + (b * 8));
    glob = (s32) &a->array2[b].x;
    glob = a[b].y;
    glob = a->array2[3].x;
    glob = (s32) &a->array2[3].x;
}
