s32 bar(f32 x);                                     /* static */
extern s32 (*_glob2)(f32);

void test(void) {
    _glob = foo;
    _glob = bar;
    _glob2 = foo;
    _glob2 = bar;
}
