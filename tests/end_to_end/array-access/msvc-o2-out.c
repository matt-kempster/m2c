void test(struct A *a, s32 b) {
    glob = (s32) a->array[b];
    glob = (s32) &a->array[b];
    glob = a->array2[b].x;
    glob = (s32) &a->array2[b].x;
    glob = *((b << 7) + (a + 0x7C));
    glob = a->array2[3].x;
    glob = (s32) &a->array2[3].x;
}
