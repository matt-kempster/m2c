s32 test(struct TestStruct *arg0, s32 arg1) {
    s32 temp_v0;

    temp_v0 = arg0->data.int_field;
    arg0->data.int_field = arg1;
    return temp_v0;
}
