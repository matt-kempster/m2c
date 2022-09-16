? func();                                           /* extern */
extern ? global_sym;
extern ? local_sym;

void test(void) {
    local_sym.unk8 = 0;
    global_sym.unk8 = 0;
    func();
}
