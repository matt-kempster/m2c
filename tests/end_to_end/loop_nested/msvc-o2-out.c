s32 test(s32 length) {
    s32 var_eax;
    s32 var_ecx;
    s32 var_edx;
    s32 var_esi;

    var_edx = 0;
    var_eax = 0;
    if (length > 0) {
        do {
            var_ecx = 0;
            var_esi = length;
loop_3:
            var_eax += var_ecx;
            var_ecx += var_edx;
            var_esi -= 1;
            if (var_esi != 0) {
                goto loop_3;
            }
            var_edx += 1;
        } while (var_edx < length);
    }
    return var_eax;
}
