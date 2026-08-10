s32 foo(s32, s32 *);                                /* static */

void test(s32 arg0, s32 *arg1) {
    s32 sp0;
    s32 temp_r4;
    s32 temp_r6;

    sp0 = arg0;
    temp_r4 = foo(arg0, &sp0);
    temp_r6 = foo(sp0, arg1);
    foo(*arg1, arg1);
}
