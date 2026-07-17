s32 inner(s32);                                     /* extern */
? outer(s32, s32);                                  /* extern */
extern s32 g;

void test(s32 arg0) {
    s32 temp_eax;

    temp_eax = g;
    outer(inner(arg0), temp_eax);
}
