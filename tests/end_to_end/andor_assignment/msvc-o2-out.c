s32 foo(s32);                                       /* static */

s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 var_eax;
    s32 var_edi;
    s32 var_esi;
    s32 var_esi_2;

    var_edi = arg0 + arg1;
    var_eax = arg1 + arg2;
    if ((var_edi == 0) && (var_eax == 0) && (var_eax = foo(var_eax), (var_eax == 0)) && (var_edi = 2, (arg3 == 0))) {
        var_esi_2 = (u8) (arg0 != 0) - 2;
    } else {
        var_esi_2 = 1;
    }
    var_esi = var_esi_2 + arg2;
    if (var_edi != 0) {
        if (var_eax != 0) {
            var_edi += var_eax;
            var_eax = foo(var_edi);
            if ((var_eax != 0) && (arg3 != 0)) {
                if (var_esi < 5) {
                    do {
                        var_esi += var_esi + 2;
                    } while (var_esi < 5);
                }
                var_esi += 5;
            }
        }
        if ((var_edi != 0) && (var_eax != 0) && (foo(var_eax + var_edi) != 0) && (arg3 != 0)) {
            if (var_esi < 5) {
                do {
                    var_esi += var_esi + 2;
                } while (var_esi < 5);
            }
            return var_esi + 5;
        }
        /* Duplicate return node #20. Try simplifying control flow for better match */
        return var_esi + 6;
    }
    return var_esi + 6;
}
