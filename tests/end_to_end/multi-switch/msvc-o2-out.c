s32 test(s32 x) {
    s32 var_ecx;

    var_ecx = x;
    switch (var_ecx) {
    case 1:
        return var_ecx * var_ecx;
    case 2:
        var_ecx -= 1;
        /* fallthrough */
    case 3:
    case 101:
    case 200:
        return (var_ecx + 1) ^ var_ecx;
    case -50:
        glob = var_ecx - 1;
        return 2;
    case 50:
    case 107:
        glob = var_ecx + 1;
        return 2;
    case 6:
    case 7:
        var_ecx *= 2;
        /* fallthrough */
    case 102:
        if (glob == 0) {
        case 103:
        case 104:
        case 105:
        case 106:
            var_ecx -= 1;
        default:
            var_ecx = (s32) (var_ecx - (var_ecx >> 0x1F)) >> 1;
        }
        glob = var_ecx;
        return 2;
    }
}
