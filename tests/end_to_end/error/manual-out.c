? test(s32 arg0) {
    s32 temp_t1;

    BREAK();
    BREAK(2);
    SYNC();
    TRAP_IF(arg0 == 0);
    TRAP_IF(arg0 != 0);
    TRAP_IF(arg0 > 0);
    TRAP_IF((u32) arg0 > 0U);
    TRAP_IF(arg0 <= 0);
    TRAP_IF((u32) arg0 <= 0U);
    TRAP_IF(arg0 == 1);
    TRAP_IF(arg0 != 2);
    TRAP_IF(arg0 < 3);
    TRAP_IF((u32) arg0 < 4U);
    TRAP_IF(arg0 >= 5);
    TRAP_IF((u32) arg0 >= 6U);
    ERROR(unknown instruction: badinstr $t0, $t0);
    temp_t1 = ERROR(unknown instruction: badinstr2 $t1, $t1);
    *NULL = (s32) (temp_t1 << temp_t1);
    *NULL = (s32) (ERROR(Read from unset register $v1) + 2);
    return ERROR(unknown instruction: badinstr3 $v0, $t2);
}
