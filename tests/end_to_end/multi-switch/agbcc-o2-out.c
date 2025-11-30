extern s32 glob;

s32 test(s32 arg0) {
    s32 var_r2;

    var_r2 = arg0;
    if (var_r2 != 0x00000032) {
        if (var_r2 <= 0x32) {
            switch (var_r2) { // switch 1; irregular
            default: // switch 1
                var_r2 *= 2;
                goto block_25;
            case 1: // switch 1
                return 1;
            case 2: // switch 1
                var_r2 = 1;
                goto block_21;
            case -50: // switch 1
                var_r2 -= 1;
                goto block_28;
            }
        } else {
            switch (var_r2) { // irregular
            case 0xC8:
            case 0x65:
            case 3: // switch 1
block_21:
                return (var_r2 + 1) ^ var_r2;
            case 0x6B:
                var_r2 = 0x0000006C;
                goto block_28;
            case 0x66:
block_25:
                if (glob == 0) {
                    var_r2 -= 1;
                    var_r2 = (s32) (var_r2 + ((u32) var_r2 >> 0x1F)) >> 1;
                }
                goto block_29;
            }
        }
    } else {
        var_r2 = 0x00000033;
block_28:
block_29:
        glob = var_r2;
        return 2;
    }
}
