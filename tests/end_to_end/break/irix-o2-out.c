extern ? D_410150;

void test(s32 arg0) {
    s32 temp_v0;
    s32 phi_v0;

    phi_v0 = 0;
    if (arg0 > 0) {
loop_2:
        temp_v0 = phi_v0 + 1;
        D_410150.unk0 = 1;
        phi_v0 = temp_v0;
        if (D_410150.unk4 == 2) {
            D_410150.unk8 = 3;
        } else {
            if (D_410150.unk8 == 2) {
                D_410150.unkC = 3;
                goto block_11;
            }
            if (D_410150.unk10 == 2) {
                D_410150.unk14 = 3;
            } else {
                if (D_410150.unk14 == 2) {
                    D_410150.unk18 = 3;
                } else {
                    D_410150.unkC = 4;
                }
block_11:
                if (temp_v0 != arg0) {
                    goto loop_2;
                }
            }
        }
    }
    D_410150.unk10 = 5;
}
