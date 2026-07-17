? f(s32);                                           /* extern */
extern s32 g;

void test(void) {
    s32 temp_call_arg;

    temp_call_arg = g;
    g = 5;
    f(temp_call_arg);
}
