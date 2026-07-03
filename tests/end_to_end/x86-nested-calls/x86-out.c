s32 _inner(s32);                                    /* extern */
? _outer(s32, s32);                                 /* extern */

void test(s32 arg0, s32 arg1) {
    _outer(_inner(arg0), arg1);
}
