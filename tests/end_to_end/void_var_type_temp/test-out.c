s32 test(void) {
    struct TestStruct *temp_t0;

    temp_t0 = *(struct TestStruct **)0x80000000;
    return temp_t0->special_field + temp_t0->base_value;
}
