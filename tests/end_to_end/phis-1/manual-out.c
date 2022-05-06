void *foo();                                        /* extern */

void test(void) {
    void *sp10C;
    void *temp_v0;

    temp_v0 = foo();
    if (*NULL == 0) {
        sp10C = temp_v0;
    }
    temp_v0->unk3 = 0;
    temp_v0->unk4 = 0;
}
