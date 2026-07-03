extern s32 _glob;

s32 test(s32 arg0) {
    s32 var_eax;

    var_eax = arg0;
    switch (var_eax) {
    case 1:
        return var_eax * var_eax;
    case 2:
        var_eax = 1;
        /* fallthrough */
    case 3:
        return var_eax * 2;
    case 4:
        _glob = var_eax + 1;
        return 2;
    case 6:
    case 7:
        _glob = var_eax * 2;
        return 2;
    default:
        _glob = (s32) (var_eax - (var_eax >> 0x1F)) >> 1;
        return 2;
    }
}
