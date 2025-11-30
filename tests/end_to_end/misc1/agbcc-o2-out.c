? bar(s32, s32, s32);                               /* static */
s32 foo(s32, s32, s32, s32);                        /* static */
extern s32 global;

s32 test(s32 arg0, s32 arg1) {
    s32 temp_r0;
    s32 temp_r2;
    s32 temp_r6;
    s32 temp_r7;

    temp_r2 = arg0 * 8;
    temp_r6 = *(global + 4 + temp_r2) + 1;
    temp_r7 = *(global + 8 + temp_r2);
    unksp0 = arg0;
    temp_r0 = foo(1, 2, temp_r6, arg1);
    if (temp_r0 != 0) {
        bar(temp_r7, temp_r0, temp_r6);
        *(arg0 + .L5.unk4) = 5;
        return temp_r0;
    }
    return 0;
}
