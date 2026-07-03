void test(s32 *arg0) {
    s32 temp_eax;

    temp_eax = *arg0;
    *arg0 = (s32) (temp_eax - (temp_eax >> 0x1F)) >> 1;
}
