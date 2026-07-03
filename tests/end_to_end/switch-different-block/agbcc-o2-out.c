extern s32 glob;

s32 test(u32 arg0) {
    s32 var_r2_2;
    u32 var_r2;

    var_r2 = arg0;
    switch (var_r2) {
    case 1:
        return var_r2 * var_r2;
    case 2:
        var_r2 -= 1;
        /* fallthrough */
    case 3:
        return var_r2 * 2;
    case 4:
        var_r2_2 = var_r2 + 1;
block_8:
        glob = var_r2_2;
        return 2;
    case 6:
    case 7:
        var_r2_2 = var_r2 * 2;
        goto block_8;
    default:
        var_r2_2 = (s32) (var_r2 + (var_r2 >> 0x1F)) >> 1;
        goto block_8;
    }
}
