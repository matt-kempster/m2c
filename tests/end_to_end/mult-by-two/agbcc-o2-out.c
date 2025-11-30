s32 __adddf3(s32, s32);                             /* extern */
s32 __addsf3(s32);                                  /* extern */
extern s32 x;

void test(void) {
    s32 temp_r1;
    s32 temp_ret;
    void *temp_r4;

    x = __addsf3(x);
    temp_r4 = .L3.unk4;
    temp_ret = __adddf3(temp_r4->unk0, temp_r4->unk4);
    temp_r1 = SECOND_REG(temp_ret);
    temp_r4->unk0 = temp_ret;
    temp_r4->unk4 = temp_r1;
}
