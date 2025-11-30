extern ? globals;

void test(s32 arg0) {
    s32 var_r0;
    s32 var_r2;

    var_r2 = 0;
loop_6:
    if (var_r2 < arg0) {
        globals.unk0 = 1;
        if (globals.unk4 == 2) {
            globals.unk8 = 3;
        } else {
            if (globals.unk8 == 2) {
                var_r0 = 3;
                goto block_4;
            }
            if (globals.unk10 == 2) {
                globals.unk14 = 3;
            } else {
                if (globals.unk14 == 2) {
                    globals.unk18 = 3;
                } else {
                    var_r0 = 4;
block_4:
                    globals.unkC = var_r0;
                }
                var_r2 += 1;
                goto loop_6;
            }
        }
    }
    globals.unk10 = 5;
}
