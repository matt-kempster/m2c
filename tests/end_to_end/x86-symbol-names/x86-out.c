? _crt();                                           /* extern */
? callee();                                         /* extern */
? _foo();                                           /* extern */
? stdcall(s32);                                     /* extern */
? foo();                                            /* extern */
static s32 data = 1;

void test(void) {
    stdcall(data);
    callee();
    _crt();
    _foo();
    foo();
}
