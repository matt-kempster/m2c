extern s32 _glob;

s32 test(s32 arg0) {
    s32 var_eax;
    u32 temp_ecx;

    var_eax = arg0;
    temp_ecx = var_eax - 1;
    switch (temp_ecx) {
    case 0:
        return var_eax * var_eax;
    case 1:
        var_eax = 1;
        /* fallthrough */
    case 2:
        return var_eax * 2;
    case 3:
        _glob = var_eax + 1;
        return 2;
    case 5:
    case 6:
        _glob = var_eax * 2;
        return 2;
    default:
        _glob = (s32) (var_eax - (var_eax >> 0x1F)) >> 1;
        return 2;
    }
}
