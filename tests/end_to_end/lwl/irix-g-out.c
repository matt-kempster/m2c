void test(void) {
    ? sp18;

    sp18 = (s32) D_400140;
    ERROR(unknown instruction: swr $at, 0x6($t6));
    func_004000B0(&sp18);
    ERROR(unknown instruction: swl $t1, ($at));
    ERROR(unknown instruction: swr $t1, 0x3($at));
    D_410160 = (?32) (unaligned s32) D_400148;
}
