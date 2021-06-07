s32 bar(f32);
extern s32 (*glob2)(f32);
void test();

void test(void) {
    glob = foo;
    glob = &bar;
    glob2 = (s32 (*)(f32)) foo;
    glob2 = &bar;
}
