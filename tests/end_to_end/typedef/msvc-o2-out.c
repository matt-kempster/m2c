s32 foo(s32, s32 *);                                /* static */

void test(s32 arg0, s32 *arg1) {
    s32 temp_edi;
    s32 temp_edi_2;

    temp_edi = foo(*arg1, arg1);
    temp_edi_2 = temp_edi + foo(arg0, arg1);
    foo(arg0, &arg0);
}
