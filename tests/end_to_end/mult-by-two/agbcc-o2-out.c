s32 __adddf3(s32, s32);                             /* extern */
s32 __addsf3(s32);                                  /* extern */
extern s32 x;
extern ? y;

void test(void) {
    s32 temp_r1;
    s32 temp_ret;

    x = __addsf3(x);
    temp_ret = __adddf3(y.unk0, y.unk4);
    temp_r1 = SECOND_REG(temp_ret);
    y.unk0 = temp_ret;
    y.unk4 = temp_r1;
}
