s32 __adddf3(s32, s32, s32, s32);                   /* extern */
s32 __floatsidf(s32);                               /* extern */
s32 __muldf3(s32, s32, s32, s32);                   /* extern */
? __truncdfsf2(s32, s32);                           /* extern */

void test(s32 arg0) {
    s32 temp_r0;
    s32 temp_r0_2;
    s32 temp_r1;
    s32 temp_ret;
    s32 temp_ret_2;
    s32 temp_ret_3;
    s32 var_r2;
    s32 var_r3;
    s32 var_r4;
    s32 var_r5;
    s32 var_r6;

    var_r6 = .L9.unk4;
    var_r5 = .L9.unk0;
    var_r4 = arg0;
    if (var_r4 != 0) {
        do {
            temp_ret = __floatsidf(var_r4);
            temp_r0 = temp_ret;
            temp_r1 = SECOND_REG(temp_ret);
            var_r3 = temp_r1;
            var_r2 = temp_r0;
            if (var_r4 < 0) {
                temp_ret_2 = __adddf3(temp_r0, temp_r1, .L9.unk8, .L9.unkC);
                var_r3 = SECOND_REG(temp_ret_2);
                var_r2 = temp_ret_2;
            }
            temp_ret_3 = __muldf3(var_r5, var_r6, var_r2, var_r3);
            temp_r0_2 = temp_ret_3;
            var_r6 = SECOND_REG(temp_ret_3);
            var_r5 = temp_r0_2;
            var_r4 -= 1;
        } while (var_r4 != 0);
    }
    __truncdfsf2(var_r5, var_r6);
}
