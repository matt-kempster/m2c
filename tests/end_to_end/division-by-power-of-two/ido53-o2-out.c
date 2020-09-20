s32 test(void *arg0) {
    s32 temp_v0;
    s32 phi_t6;

    temp_v0 = *arg0;
    phi_t6 = temp_v0 >> 1;
    if (temp_v0 < 0) {
        phi_t6 = (s32) (temp_v0 + 1) >> 1;
    }
    *arg0 = phi_t6;
    return temp_v0;
}
