void test(s32 *arg0) {
    MIPS2C_ERROR(unknown instruction: addze $r0, $r0);
    *arg0 = MIPS2C_ERROR(unknown instruction: subfc $r0, $r0, $r4);
}
