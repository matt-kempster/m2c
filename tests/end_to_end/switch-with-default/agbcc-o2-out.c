extern s32 glob;

s32 test(s32 arg0) {
    s32 var_r4;

    var_r4 = arg0;
    switch (var_r4) {                               /* irregular */
    case 9:
        break;
    case 2:
        var_r4 = 1;
        /* fallthrough */
    case 3:
        glob += 1;
        break;
    case 13:
        var_r4 = 0x1A;
        break;
    default:
        var_r4 = (s32) (var_r4 + ((u32) var_r4 >> 0x1F)) >> 1;
        break;
    }
    test(glob);
    if (glob == 0) {
        glob = var_r4;
    }
    return 2;
}
