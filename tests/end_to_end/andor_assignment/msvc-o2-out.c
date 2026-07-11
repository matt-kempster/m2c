s32 test(s32 a, s32 b, s32 c, s32 d) {
    s32 var_eax;
    s32 var_edi;
    s32 var_esi;
    s32 var_esi_2;

    var_edi = a + b;
    var_eax = b + c;
    if ((var_edi == 0) && (var_eax == 0) && (var_eax = foo(var_eax), (var_eax == 0)) && (var_edi = 2, (d == 0))) {
        var_esi_2 = (u8) (a != 0) - 2;
    } else {
        var_esi_2 = 1;
    }
    var_esi = var_esi_2 + c;
    if (var_edi != 0) {
        if (var_eax != 0) {
            var_edi += var_eax;
            var_eax = foo(var_edi);
            if ((var_eax != 0) && (d != 0)) {
                if (var_esi < 5) {
                    do {
                        var_esi += var_esi + 2;
                    } while (var_esi < 5);
                }
                var_esi += 5;
            }
        }
        if ((var_edi != 0) && (var_eax != 0) && (foo(var_eax + var_edi) != 0) && (d != 0)) {
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
