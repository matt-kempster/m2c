? __adddf3(s32, s32, s32, s32);                     /* extern */
s32 __addsf3(s32, s32);                             /* extern */
s32 __extendsfdf2(s32);                             /* extern */
s32 __fixsfsi();                                    /* extern */
s32 __floatsisf(s32);                               /* extern */
s32 __truncdfsf2();                                 /* extern */
extern s32 globalf;
extern s32 globali;

void test(s32 arg1, s32 arg2, s32 arg3) {
    s32 temp_r0;
    s32 temp_r6;
    s32 temp_r7;
    s32 temp_ret;
    s32 temp_ret_2;
    s32 temp_ret_3;
    s32 var_r0;

    globali = __fixsfsi();
    globalf = __floatsisf(arg1);
    temp_r7 = arg3 + 3;
    temp_ret = __extendsfdf2(arg2);
    __adddf3(temp_ret, SECOND_REG(temp_ret), 0, .L6.unkC);
    __truncdfsf2();
    temp_ret_2 = __extendsfdf2();
    __adddf3(temp_ret_2, SECOND_REG(temp_ret_2), .L6.unk10, .L6.unk14);
    __addsf3(__truncdfsf2(), .L6.unk18);
    temp_ret_3 = __extendsfdf2();
    __adddf3(temp_ret_3, SECOND_REG(temp_ret_3), .L6.unk1C, .L6.unk20);
    temp_r6 = __truncdfsf2();
    if (temp_r7 >= 0) {
        var_r0 = __floatsisf(temp_r7);
    } else {
        temp_r0 = __floatsisf((1 & temp_r7) | ((u32) temp_r7 >> 1));
        var_r0 = __addsf3(temp_r0, temp_r0);
    }
    __addsf3(var_r0, temp_r6);
}
