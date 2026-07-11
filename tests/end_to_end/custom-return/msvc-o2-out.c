s32 test(void) {
    u16 var_eax;

    var_eax = foo(1);
    if (var_eax == 0) {
        if (_glob != 0x7B) {
            return (s32) foo(2);
        }
        var_eax = foo(3);
        /* Duplicate return node #4. Try simplifying control flow for better match */
        return (s32) var_eax;
    }
    return (s32) var_eax;
}
