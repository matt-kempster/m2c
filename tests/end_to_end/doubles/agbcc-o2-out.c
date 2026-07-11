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
    s32 temp_r0;
    s32 temp_r1;
    s32 temp_r4;
    s32 temp_r5;
    s32 temp_ret;
    s32 temp_ret_2;
    s32 temp_ret_3;
    s32 temp_ret_4;
    s32 temp_ret_5;
    s32 var_r1;

    sp14 = arg3;
    temp_ret = __floatsidf(arg2);
    temp_ret_2 = __muldf3(arg0, arg1, temp_ret, SECOND_REG(temp_ret));
    temp_r5 = SECOND_REG(temp_ret_2);
    temp_r4 = temp_ret_2;
    temp_ret_3 = __divdf3(arg0, arg1, sp14, arg4);
    temp_ret_4 = __adddf3(temp_r4, temp_r5, temp_ret_3, SECOND_REG(temp_ret_3));
    temp_ret_5 = __subdf3(temp_ret_4, SECOND_REG(temp_ret_4), 0x401C0000, 0);
    temp_r0 = temp_ret_5;
    temp_r1 = SECOND_REG(temp_ret_5);
    if ((__ltdf2(temp_r0, temp_r1, sp14, arg4) < 0) || (__eqdf2(temp_r0, temp_r1, sp14, arg4) == 0) || (__gtdf2(temp_r0, temp_r1, 0x40220000, 0) > 0)) {
        var_r1 = 0x40140000;
    } else {
        var_r1 = 0x40180000;
    }
    global.unk0 = var_r1;
    global.unk4 = 0;
    M2C_ERROR(/* Read from unset register $r3 */)(var_r1, 0);
}
