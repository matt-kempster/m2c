s8 test(s8 *ptr) {
    return ptr->unk3;
}

s16 test_loadw_disp(s16 *ptr) {
    return ptr->unk6;
}

s32 test_loadl_disp(s32 *ptr) {
    return ptr->unkC;
}

void test_storeb_disp(s8 *ptr, s8 value) {
    ptr->unk3 = value;
}

void test_storew_disp(s16 *ptr, s16 value) {
    ptr->unk6 = value;
}

void test_storel_disp(s32 *ptr, s32 value) {
    ptr->unkC = value;
}

s8 test_loadb_indexed(s8 *ptr, u32 index) {
    return ptr[index];
}

s16 test_loadw_indexed(s16 *ptr, u32 index) {
    return ptr[index];
}

s32 test_loadl_indexed(s32 *ptr, u32 index) {
    return ptr[index];
}

void test_storeb_indexed(s8 *ptr, u32 index, s8 value) {
    ptr[index] = value;
}

void test_storew_indexed(s16 *ptr, u32 index, s16 value) {
    ptr[index] = value;
}

void test_storel_indexed(s32 *ptr, u32 index, s32 value) {
    ptr[index] = value;
}
