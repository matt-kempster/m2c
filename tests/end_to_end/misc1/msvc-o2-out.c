? bar(s32, s32, s32);                               /* static */
s32 foo(?, ?, s32, s32, s32);                       /* static */
extern s32 _global;
extern ? _global2;

s32 test(s32 arg0, s32 arg1) {
    s32 temp_eax;
    s32 temp_ebp;
    s32 temp_esi;

    temp_ebp = *(_global + ((arg0 * 8) + 8));
    temp_esi = *(_global + ((arg0 * 8) + 4)) + 1;
    temp_eax = foo(1, 2, temp_esi, arg1, arg0);
    if (temp_eax == 0) {
        return temp_eax;
    }
    bar(temp_ebp, temp_eax, temp_esi);
    *(arg0 + &_global2) = 5;
    return temp_eax;
}
