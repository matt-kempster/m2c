f32 test(s32 *arg0) {
    s32 *spC;
    s32 *sp8;
    s32 *sp4;
    s32 temp_a1;

    temp_a1 = *arg0;
    if (temp_a1 != 0) {
        if (temp_a1 != 1) {
            if (temp_a1 != 2) {
                return 0.0f;
            }
            sp4 = arg0;
            return sp4->unk4 + sp4->unkC;
        }
        sp8 = arg0;
        return sp8->unk4 + sp8->unkC;
    }
    spC = arg0;
    return spC->unk4 + spC->unkC;
}
