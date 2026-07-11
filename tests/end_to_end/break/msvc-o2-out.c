void test(s32 lim) {
    s32 var_eax;

    var_eax = 0;
    if (lim > 0) {
loop_3:
        globals[0] = 1;
        if (globals[1] != 2) {
            if (globals[2] == 2) {
                globals[3] = 3;
                goto block_10;
            }
            if (globals[4] != 2) {
                if (globals[5] == 2) {
                    globals[6] = 3;
                } else {
                    globals[3] = 4;
                }
block_10:
                var_eax += 1;
                if (var_eax >= lim) {
                    globals[4] = 5;
                    return;
                }
                goto loop_3;
            }
            globals[5] = 3;
            /* Duplicate return node #14. Try simplifying control flow for better match */
            globals[4] = 5;
            return;
        }
        globals[2] = 3;
        globals[4] = 5;
        return;
    }
    globals[4] = 5;
}
