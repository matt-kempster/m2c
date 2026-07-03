s32 test(s32 *arg0, s32 *arg1) {
    s32 temp_eax;

    temp_eax = *arg0;
    if (temp_eax != 8) {
        if (temp_eax == 0xF) {
            *arg1 -= 0xF;
            return 0;
        }
        /* Duplicate return node #4. Try simplifying control flow for better match */
        return 0;
    }
    *arg1 += 8;
    return 0;
}
