s32 test(s32 arg0) {
    s32 temp_t6;
    s32 phi_s0;

    phi_s0 = arg0;
    goto block_2;
loop_1:
    func_004000DC(temp_t6);
    phi_s0 = temp_t6 + 1;
block_2:
    func_004000DC(phi_s0);
    temp_t6 = phi_s0 * 2;
    if (temp_t6 < 4) {
        goto loop_1;
    }
    return temp_t6;
}
