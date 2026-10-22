? bar(void *);                                      /* extern */
extern ? D_410000;

void test(s32 arg0) {
    bar(arg0 + &D_410000);
}
