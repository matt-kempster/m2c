void test(void) {
    glob = foo;
    glob = &bar;
    glob2 = (s32 (*)(f32)) foo;
    glob2 = &bar;
}
