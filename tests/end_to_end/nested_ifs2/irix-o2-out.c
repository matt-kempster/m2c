? func_004000F0(?, s32);                            /* static */
void test(s32 arg0);                                /* static */

void test(s32 arg0) {
    s32 temp_a1;

    temp_a1 = arg0;
    if (arg0 == 7) {
        arg0 = temp_a1;
        func_004000F0(1, temp_a1);
        if (arg0 == 8) {
            func_004000F0(2, arg0);
        }
        func_004000F0(3);
        return;
    }
    func_004000F0(4, temp_a1);
}
