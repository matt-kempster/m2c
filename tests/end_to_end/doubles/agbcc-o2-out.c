s32 __adddf3(s32, s32, s32, s32);                   /* extern */
s32 __divdf3(s32, s32, s32, s32);                   /* extern */
s32 __eqdf2(s32, s32, s32, s32);                    /* extern */
s32 __floatsidf(s32);                               /* extern */
s32 __gtdf2(s32, s32, s32, s32);                    /* extern */
s32 __ltdf2(s32, s32, s32, s32);                    /* extern */
s32 __muldf3(s32, s32, s32, s32);                   /* extern */
s32 __subdf3(s32, s32, s32, s32);                   /* extern */
extern ? global;

void test(s32 arg0, s32 arg1, s32 arg2, s32 arg3, s32 arg4) {
    s32 sp14;
    ? *var_r1;
    s32 temp_r0;
    s32 temp_r1;
    s32 temp_r4;
    s32 temp_r5;
    s32 temp_ret;
    s32 temp_ret_2;
    s32 temp_ret_3;
    s32 temp_ret_4;
    s32 temp_ret_5;
    s32 var_r2;

    sp14 = arg3;
    temp_ret = __floatsidf(arg2);
    temp_ret_2 = __muldf3(arg0, arg1, temp_ret, SECOND_REG(temp_ret));
    temp_r5 = SECOND_REG(temp_ret_2);
    temp_r4 = temp_ret_2;
    temp_ret_3 = __divdf3(arg0, arg1, sp14, arg4);
    temp_ret_4 = __adddf3(temp_r4, temp_r5, temp_ret_3, SECOND_REG(temp_ret_3));
    temp_ret_5 = __subdf3(temp_ret_4, SECOND_REG(temp_ret_4), .L6.unk0, .L6.unk4);
    temp_r0 = temp_ret_5;
    temp_r1 = SECOND_REG(temp_ret_5);
    if ((__ltdf2(temp_r0, temp_r1, sp14, arg4) < 0) || (__eqdf2(temp_r0, temp_r1, sp14, arg4) == 0) || (__gtdf2(temp_r0, temp_r1, .L6.unk8, .L6.unkC) > 0)) {
        var_r1 = .L6.unk10;
        var_r2 = .L6.unk14;
    } else {
        var_r2 = .L8.unk4;
        var_r1 = &global;
    }
    .L8.unk8->unk0 = var_r1;
    .L8.unk8->unk4 = var_r2;
    M2C_ERROR(/* Read from unset register $r3 */)(var_r1, var_r2);
}
