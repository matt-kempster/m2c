extern s32 D_4100F0;

void test(void) {
    // Flowgraph is not reducible, falling back to gotos-only mode. (Are there infinite loops?)
    D_4100F0 = 1;
loop_1:
    goto loop_1;
}
