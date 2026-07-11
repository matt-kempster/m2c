void test(void) {
    s32 temp_eax;

    temp_eax = foo();
    global = 1;
    bar(temp_eax);
}
