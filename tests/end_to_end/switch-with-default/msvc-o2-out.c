s32 test(s32 x) {
    s32 var_esi;

    var_esi = x;
    switch (var_esi) {
    case 9:
        break;
    case 2:
        var_esi -= 1;
        /* fallthrough */
    case 3:
        _glob += 1;
        break;
    case 13:
        var_esi *= 2;
        break;
    default:
    case 4:
    case 5:
    case 6:
    case 7:
    case 8:
    case 10:
    case 11:
    case 12:
        var_esi = (s32) (var_esi - (var_esi >> 0x1F)) >> 1;
        break;
    }
    test(_glob);
    if (_glob == 0) {
        _glob = var_esi;
    }
    return 2;
}
