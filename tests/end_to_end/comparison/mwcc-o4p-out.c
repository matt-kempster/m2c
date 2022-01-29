s32 test(s32 arg0, s32 arg1, s32 arg2) {
    MIPS2C_ERROR(unknown instruction: addic $r0, $r5, -0x1);
    *NULL = (u32) MIPS2C_ERROR(unknown instruction: cntlzw $r0, $r0) >> 5U;
    MIPS2C_ERROR(unknown instruction: eqv $r0, $r4, $r3);
    *NULL = (u32) MIPS2C_ERROR(unknown instruction: subfe $r5, $r0, $r5);
    MIPS2C_ERROR(unknown instruction: subfc $r5, $r4, $r3);
    *NULL = (u32) (MIPS2C_ERROR(unknown instruction: addze $r0, $r0) & 1);
    MIPS2C_ERROR(unknown instruction: subfc $r3, $r3, $r4);
    *NULL = (u32) MIPS2C_ERROR(unknown instruction: adde $r5, $r5, $r6);
    MIPS2C_ERROR(unknown instruction: addic $r0, $r3, -0x1);
    *NULL = (u32) ((u32) MIPS2C_ERROR(unknown instruction: cntlzw $r0, $r0) >> 5U);
    *NULL = (u32) MIPS2C_ERROR(unknown instruction: subfe $r0, $r0, $r3);
    return -arg1;
}
