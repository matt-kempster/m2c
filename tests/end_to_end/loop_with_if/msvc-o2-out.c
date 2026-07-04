s32 test(s32 length) {
    s32 var_eax;

    var_eax = 0;
    if (length > 0) {
        do {
            if (var_eax == 5) {
                var_eax = 0xA;
            } else {
                var_eax += 4;
            }
        } while (var_eax < length);
    }
    return var_eax;
}
