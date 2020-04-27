? test(void) {
    s32 temp_t1;

    ERROR(unknown instruction: break 0x1);
    ERROR(unknown instruction: break 0x2);
    ERROR(unknown instruction: badinstr $t0, $t0);
    temp_t1 = ERROR(unknown instruction: badinstr2 $t1, $t1);
    *NULL = (s32) (temp_t1 << temp_t1);
    return ERROR(unknown instruction: badinstr3 $v0, $t2);
}
