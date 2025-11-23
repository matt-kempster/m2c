Warning: missing "jr $ra" in last block of test (jumptarget_label).

extern s32 more special;
static s16 special !@#$%^chars[2] = { 0, 0x1234 };  /* const */

void *test(void) {
    return &special !@#$%^chars[0x1233FFFF] + more special;
}
