f32 test(void *arg0) {
    s32 temp_v0;

    temp_v0 = arg0->unk0;
    if (temp_v0 != 0) {
        if (temp_v0 != 1) {
            if (temp_v0 != 2) {
                return 0.0f;
            }
            return arg0->unk4 + arg0->unkC;
        }
        return arg0->unk4 + arg0->unkC;
    }
    return arg0->unk4 + arg0->unkC;
}
