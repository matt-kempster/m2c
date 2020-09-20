s32 test(void *arg0) {
    s32 temp_v0;
    s32 temp_v0_2;
    s32 phi_at;

    temp_v0_2 = *arg0;
    phi_at = temp_v0_2;
    if (temp_v0_2 < 0) {
        phi_at = temp_v0_2 + 1;
    }
    temp_v0 = phi_at >> 1;
    *arg0 = temp_v0;
    return temp_v0;
}
