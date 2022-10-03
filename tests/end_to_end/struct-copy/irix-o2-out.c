extern ? D_410150;
extern s32 D_4102E0;

void test(void *arg0, ? *arg1) {
    ? *var_t4;
    ? *var_t6;
    ? temp_at_2;
    s32 *var_t7;
    s32 temp_at;
    void *var_t5;

    var_t7 = &D_4102E0;
    var_t6 = &D_410150;
    do {
        temp_at = *var_t7;
        var_t7 += 0xC;
        var_t6 += 0xC;
        var_t6->unk-C = temp_at;
        var_t6->unk-8 = (s32) var_t7->unk-8;
        var_t6->unk-4 = (s32) var_t7->unk-4;
    } while (var_t7 != (&D_4102E0 + 0x18C));
    var_t4 = arg1;
    var_t5 = arg0;
    var_t6->unk0 = (s32) var_t7->unk0;
    do {
        temp_at_2 = (unaligned s32) *var_t4;
        var_t4 += 0xC;
        var_t5 += 0xC;
        var_t5->unk-C = (unaligned s32) temp_at_2;
        var_t5->unk-8 = (unaligned s32) var_t4->unk-8;
        var_t5->unk-4 = (unaligned s32) var_t4->unk-4;
    } while (var_t4 != (arg1 + 0x60));
    var_t5->unk0 = (unaligned s32) var_t4->unk0;
}
