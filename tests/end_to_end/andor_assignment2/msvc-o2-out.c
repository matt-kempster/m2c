s32 foo(s32);                                       /* static */

s32 test(s32 arg0, s32 arg1, s32 arg2, s32 arg3) {
    s32 temp_eax;
    s32 temp_eax_2;
    s32 temp_eax_3;
    s32 temp_ebp;
    s32 temp_ebx;
    s32 temp_edi;
    s32 temp_esi;

    temp_eax = arg0 + arg1;
    temp_ebx = arg1 + arg2;
    temp_ebp = arg2 + arg3;
    if ((temp_eax != 0) && (temp_ebx != 0) && (temp_ebp != 0)) {
        temp_eax_2 = foo(temp_eax + arg0);
        if (temp_eax_2 > 0xA) {
            temp_esi = foo(temp_eax_2 + arg1);
            temp_edi = foo(temp_ebx + arg2);
            temp_eax_3 = foo(temp_ebp + arg3);
            if ((temp_esi != 0) && (temp_edi != 0) && (temp_eax_3 != 0)) {
                return 1;
            }
        }
    }
    return 0;
}
