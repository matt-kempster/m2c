void test(void) {
    s32 *sp1C;
    struct A *sp18;

    foo(&sp1C, &sp18);
    foo((s32 **) ((intptr_t) &sp1C & ~3), &sp18);
}
