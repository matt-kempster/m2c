s32 test(s32 arg0, s32 arg1) {
    s32 temp_eax;
    s32 temp_ebp;
    s32 temp_esi;

    temp_ebp = global->b[arg0].b;
    temp_esi = global->b[arg0].a + 1;
    temp_eax = foo(1, 2, temp_esi, arg1, arg0);
    if (temp_eax == 0) {
        return temp_eax;
    }
    bar(temp_ebp, temp_eax, temp_esi);
    global2[arg0] = 5;
    return temp_eax;
}
