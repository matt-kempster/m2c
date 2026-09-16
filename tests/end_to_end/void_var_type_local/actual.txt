s32 test(void) {
    struct TestStruct *sp18;

    sp18 = *(struct TestStruct **)0x80000000;
    return sp18->field_0x00;
}
