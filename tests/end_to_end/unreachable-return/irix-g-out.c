extern s32 D_4100E0;

void test(void) {
    // Flowgraph is not reducible, falling back to gotos-only mode. (Are there infinite loops?)
loop_0:
    D_4100E0 = 1;
    goto loop_0;
}
