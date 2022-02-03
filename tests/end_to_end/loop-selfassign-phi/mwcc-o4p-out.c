s32 foo();                                          /* static */

void test(s32 arg0) {
    s32 phi_r3;

    phi_r3 = arg0;
loop_1:
    if (phi_r3 <= 2) {
        goto loop_1;
    }
    phi_r3 = foo();
    goto loop_1;
}
