s32 __extendsfdf2();                                /* extern */
s32 __gedf2(s32, s32, s32, s32);                    /* extern */
s32 __gesf2(s32, s32);                              /* extern */
s32 __negdf2(s32, s32);                             /* extern */
s32 __negsf2(s32);                                  /* extern */
? __truncdfsf2();                                   /* extern */
? sqrt(s32, s32);                                   /* extern */
? sqrtf(s32);                                       /* extern */

void test(s32 arg0) {
    s32 temp_r0;
    s32 temp_r0_2;
    s32 temp_r1;
    s32 temp_ret;
    s32 temp_ret_2;
    s32 var_r4;
    s32 var_r4_2;
    s32 var_r5;

    var_r4 = arg0;
    if (__gesf2(arg0, 0) < 0) {
        var_r4 = __negsf2(var_r4);
    }
    sqrtf(var_r4);
    temp_ret = __extendsfdf2();
    temp_r0 = temp_ret;
    temp_r1 = SECOND_REG(temp_ret);
    var_r5 = temp_r1;
    var_r4_2 = temp_r0;
    if (__gedf2(temp_r0, temp_r1, .L5.unk4, .L5.unk8) < 0) {
        temp_ret_2 = __negdf2(var_r4_2, var_r5);
        temp_r0_2 = temp_ret_2;
        var_r5 = SECOND_REG(temp_ret_2);
        var_r4_2 = temp_r0_2;
    }
    sqrt(var_r4_2, var_r5);
    __truncdfsf2();
}
