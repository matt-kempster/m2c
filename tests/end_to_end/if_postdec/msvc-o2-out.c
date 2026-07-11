s32 test(void) {
    s32 temp_ecx;
    s32 var_eax;

    temp_ecx = _glob;
    _glob -= 1;
    var_eax = 4;
    if (temp_ecx >= 1) {
        var_eax = 6;
    }
    return var_eax;
}
