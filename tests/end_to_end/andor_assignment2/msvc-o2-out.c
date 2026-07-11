s32 test(s32 a, s32 b, s32 c, s32 d) {
    s32 temp_eax;
    s32 temp_eax_2;
    s32 temp_eax_3;
    s32 temp_ebp;
    s32 temp_ebx;
    s32 temp_edi;
    s32 temp_esi;

    temp_eax = a + b;
    temp_ebx = b + c;
    temp_ebp = c + d;
    if ((temp_eax != 0) && (temp_ebx != 0) && (temp_ebp != 0)) {
        temp_eax_2 = foo(temp_eax + a);
        if (temp_eax_2 > 0xA) {
            temp_esi = foo(temp_eax_2 + b);
            temp_edi = foo(temp_ebx + c);
            temp_eax_3 = foo(temp_ebp + d);
            if ((temp_esi != 0) && (temp_edi != 0) && (temp_eax_3 != 0)) {
                return 1;
            }
        }
    }
    return 0;
}
