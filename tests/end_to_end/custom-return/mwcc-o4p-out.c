u16 foo(?);                                         /* static */

u16 test(void) {
    u16 temp_r3;

    temp_r3 = foo(1);
    if (temp_r3 != 0U) {
        return temp_r3;
    }
    if ((s32) *NULL != 0x7B) {
        return foo(2);
    }
    return foo(3);
}
