f32 bar(); // extern

f32 test(void) {
    f32 temp_f0;
    f32 phi_f0;

    // Flowgraph is not reducible, falling back to gotos-only mode.
    temp_f0 = bar();
    phi_f0 = temp_f0;
    phi_f0 = temp_f0;
    if (4 == 0) {
        goto block_3;
    }
block_1:
    if (1 == 0) {
        goto block_3;
    }
    bar();
    phi_f0 = 2.0f;
block_3:
    if ((1 < 1) < 0) {
        goto block_1;
    }
    return phi_f0;
}
