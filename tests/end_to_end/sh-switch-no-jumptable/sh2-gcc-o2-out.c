s32 test(s32 value) {
    s32 var_r0;

    if (value != 1) {
        if (value <= 1) {
            var_r0 = 0xB;
            if (value != 0) {
                return -7;
            }
            /* Duplicate return node #7. Try simplifying control flow for better match */
            return var_r0;
        }
        var_r0 = 0x13;
        if (value != 2) {
            return -7;
        }
        /* Duplicate return node #7. Try simplifying control flow for better match */
        return var_r0;
    }
    var_r0 = 0x2A;
    return var_r0;
}
