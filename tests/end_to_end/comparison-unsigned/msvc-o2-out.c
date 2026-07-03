extern s32 _global;

void test(u32 arg0, u32 arg1) {
    _global = arg0 < arg1;
    _global = arg1 >= arg0;
}
