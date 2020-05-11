void test(void) {
    ? sp18;

    sp18 = (s32) D_400130;
    ERROR(unknown instruction: swr $at, 0x6($a0));
    func_004000B0(&sp18);
    D_410141 = (unaligned s32) D_410149;
    ERROR(unknown instruction: swr $t0, 0x3($at));
    D_410150 = (?32) (unaligned s32) D_400138;
}
