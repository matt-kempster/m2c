void test(struct A *a, s32 b) {
    _glob = (s32) a->array[b];
    _glob = (s32) &a->array[b];
    _glob = a->array2[b].x;
    _glob = (s32) &a->array2[b].x;
    _glob = *((b << 7) + (a + 0x7C));
    _glob = a->array2[3].x;
    _glob = (s32) &a->array2[3].x;
}
