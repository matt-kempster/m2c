u16 foo(?);                                         /* static */
extern s32 _glob;

u16 test(void) {
    u16 var_eax;

    var_eax = foo(1);
    if (var_eax == 0) {
        if (_glob != 0x7B) {
            return foo(2);
        }
        var_eax = foo(3);
        /* Duplicate return node #4. Try simplifying control flow for better match */
        return var_eax;
    }
    return var_eax;
}
