s32 inner(s32);                                     /* extern */
? outer(s32, s32);                                  /* extern */

void test(s32 arg0, s32 arg1) {
    outer(inner(arg0), arg1);
}
