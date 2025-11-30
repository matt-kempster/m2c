s32 __addsf3(s32, s32);                             /* extern */
s32 __fixsfsi();                                    /* extern */
s32 __floatsisf(s32);                               /* extern */
extern s32 globalf;
extern s32 globali;

void test(s32 arg1, s32 arg2, s32 arg3) {
    s32 temp_r0;
    s32 temp_r6;
    s32 temp_r7;
    s32 var_r0;

    globali = __fixsfsi();
    globalf = __floatsisf(arg1);
    temp_r6 = arg3 + 3;
    temp_r7 = __addsf3(arg2, 0x40A00000);
    if (temp_r6 >= 0) {
        var_r0 = __floatsisf(temp_r6);
    } else {
        temp_r0 = __floatsisf((1 & temp_r6) | ((u32) temp_r6 >> 1));
        var_r0 = __addsf3(temp_r0, temp_r0);
    }
    __addsf3(var_r0, temp_r7);
}
