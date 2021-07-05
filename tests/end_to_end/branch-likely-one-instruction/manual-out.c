s32 test(void) {
    s32 phi_a0;

    phi_a0 = *(s32 *)0x8009A600 == 3;
    if (*(s8 *)0x800B0F15 == 5) {
        phi_a0 = 1;
    }
    return phi_a0;
}
