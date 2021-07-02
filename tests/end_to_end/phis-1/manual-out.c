void *foo(); // extern
void test(); // static

void test(void) {
    void *sp10C;
    void *temp_ret;

    temp_ret = foo();
    if (*NULL == 0) {
        sp10C = temp_ret;
    }
    temp_ret->unk3 = (u8)0;
    temp_ret->unk4 = (u8)0;
}
