static ? globals;

void test(s32 arg0) {
    s32 temp_ctr;
    s32 phi_ctr;

    phi_ctr = arg0;
    if (arg0 > 0) {
loop_1:
        globals.unk0 = 1;
        if ((s32) globals.unk4 == 2) {
            globals.unk8 = 3;
            globals.unk10 = 5;
            return;
        }
        if ((s32) globals.unk8 == 2) {
            globals.unkC = 3;
            goto block_10;
        }
        if ((s32) globals.unk10 == 2) {
            globals.unk14 = 3;
            globals.unk10 = 5;
            return;
        }
        if ((s32) globals.unk14 == 2) {
            globals.unk18 = 3;
        } else {
            globals.unkC = 4;
        }
block_10:
        temp_ctr = phi_ctr - 1;
        phi_ctr = temp_ctr;
        if (temp_ctr == 0) {
            /* Duplicate return node #11. Try simplifying control flow for better match */
            globals.unk10 = 5;
            return;
        }
        goto loop_1;
    }
    globals.unk10 = 5;
}
