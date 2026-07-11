void test(s32 lim) {
    s32 var_eax;

    var_eax = 0;
    if (lim > 0) {
loop_3:
        _globals[0] = 1;
        if (_globals[1] != 2) {
            if (_globals[2] == 2) {
                _globals[3] = 3;
                goto block_10;
            }
            if (_globals[4] != 2) {
                if (_globals[5] == 2) {
                    _globals[6] = 3;
                } else {
                    _globals[3] = 4;
                }
block_10:
                var_eax += 1;
                if (var_eax >= lim) {
                    _globals[4] = 5;
                    return;
                }
                goto loop_3;
            }
            _globals[5] = 3;
            /* Duplicate return node #14. Try simplifying control flow for better match */
            _globals[4] = 5;
            return;
        }
        _globals[2] = 3;
        _globals[4] = 5;
        return;
    }
    _globals[4] = 5;
}
