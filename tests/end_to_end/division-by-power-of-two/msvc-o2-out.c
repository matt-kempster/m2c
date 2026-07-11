void test(s32 *ptr) {
    s32 temp_eax;

    temp_eax = *ptr;
    *ptr = (s32) (temp_eax - (temp_eax >> 0x1F)) >> 1;
}
