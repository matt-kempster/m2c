s32 test(s32 a, s32 b, s32 c, s32 d) {
    s32 var_eax;
    s32 var_ecx;

    var_eax = 0;
    if (a != 0) {
loop_1:
        if (b != 0) {
block_3:
            var_eax += 1;
            goto loop_1;
        }
        if (c != 0) {
            goto block_3;
        }
    }
loop_4:
    if (a != 0) {
block_7:
        var_eax += 1;
        goto loop_4;
    }
    if ((b != 0) && (c != 0)) {
        goto block_7;
    }
    var_ecx = 0;
loop_9:
    if (b != 0) {
        var_eax += 1;
        var_ecx += c + d;
        if (var_ecx < 0xA) {
            goto loop_9;
        }
    }
    return var_eax;
}
