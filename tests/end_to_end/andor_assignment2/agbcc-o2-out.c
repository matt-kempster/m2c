s32 foo(s32);                                       /* static */

s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 temp_r0;
    s32 temp_r0_2;
    s32 temp_r1;
    s32 temp_r4;
    s32 temp_r4_2;
    s32 temp_r6;
    s32 temp_r6_2;

    temp_r4 = arg0 + arg1;
    temp_r6 = arg1 + arg2;
    temp_r1 = arg2 + arg3;
    if ((temp_r4 != 0) && (temp_r6 != 0) && (temp_r1 != 0)) {
        temp_r0 = foo(temp_r4 + arg0);
        if (temp_r0 > 0xA) {
            temp_r4_2 = foo(temp_r0 + arg1);
            temp_r6_2 = foo(temp_r6 + arg2);
            temp_r0_2 = foo(temp_r1 + arg3);
            if ((temp_r4_2 != 0) && (temp_r6_2 != 0) && (temp_r0_2 != 0)) {
                return 1;
            }
        }
    }
    return 0;
}
