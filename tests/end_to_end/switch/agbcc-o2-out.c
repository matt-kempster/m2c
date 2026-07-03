extern s32 glob;

s32 test(u32 arg0) {
    s32 var_r2_2;
    u32 var_r2;

    var_r2 = arg0;
    switch (var_r2) {
    case 0x1:
        return var_r2 * var_r2;
    case 0x2:
        var_r2 -= 1;
        /* fallthrough */
    case 0x3:
        return var_r2 * 2;
    case 0x4:
        var_r2_2 = var_r2 + 1;
block_8:
        glob = var_r2_2;
        return 2;
    case 0x6:
    case 0x7:
        var_r2_2 = var_r2 * 2;
        goto block_8;
    default:
        var_r2_2 = (s32) (var_r2 + (var_r2 >> 0x1F)) >> 1;
        goto block_8;
    }
}
