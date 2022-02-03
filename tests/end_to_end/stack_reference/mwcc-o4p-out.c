s32 test(s32 arg0, s32 arg1) {
    s32 sp8;
    s32 spC;
    s32 *temp_r5;
    s32 temp_r0;

    temp_r5 = &sp8;
    sp8 = arg0;
    temp_r0 = &spC - temp_r5;
    spC = arg1;
    return MIPS2C_ERROR(unknown instruction: addze $r3, $r3);
}
