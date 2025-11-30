s32 __adddf3(s32, s32, s32, s32);                   /* extern */
s32 __addsf3(s32, s32);                             /* extern */
s32 __fixunsdfsi(s32, s32);                         /* extern */
s32 __fixunssfsi(s32);                              /* extern */
s32 __floatsidf(s32);                               /* extern */
s32 __floatsisf(s32);                               /* extern */
extern s32 u;

void test(void) {
    s32 *temp_r7;
    s32 temp_r0;
    s32 temp_r4;
    s32 temp_ret;
    s32 temp_ret_2;
    s32 var_r0;
    s32 var_r0_2;
    s32 var_r1;
    void *temp_r6;

    temp_r7 = .L7.unk4;
    u = __fixunssfsi(*temp_r7);
    temp_r6 = .L7.unk8;
    u = __fixunsdfsi(temp_r6->unk0, temp_r6->unk4);
    temp_r4 = u;
    temp_ret = __floatsidf(temp_r4);
    var_r0 = temp_ret;
    var_r1 = SECOND_REG(temp_ret);
    if (temp_r4 < 0) {
        temp_ret_2 = __adddf3(var_r0, var_r1, .L7.unkC, .L7.unk10);
        var_r0 = temp_ret_2;
        var_r1 = SECOND_REG(temp_ret_2);
    }
    temp_r6->unk0 = var_r0;
    temp_r6->unk4 = var_r1;
    if ((s32) u >= 0) {
        var_r0_2 = __floatsisf(u);
    } else {
        temp_r0 = __floatsisf((1 & u) | ((u32) u >> 1));
        var_r0_2 = __addsf3(temp_r0, temp_r0);
    }
    *temp_r7 = var_r0_2;
}
