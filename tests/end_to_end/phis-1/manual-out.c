void *foo(); // extern

void test(void) {
    void *sp10C;
    void *temp_ret;

    temp_ret = foo();
    if (*NULL == 0) {
        sp10C = temp_ret;
    }
    temp_ret->unk3 = 0;
    temp_ret->unk4 = 0;
}
