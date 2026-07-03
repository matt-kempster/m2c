extern ? _globals;

void test(s32 arg0) {
    s32 var_eax;

    var_eax = 0;
    if (arg0 > 0) {
loop_3:
        _globals.unk0 = 1;
        if (_globals.unk4 != 2) {
            if (_globals.unk8 == 2) {
                _globals.unkC = 3;
                goto block_10;
            }
            if (_globals.unk10 != 2) {
                if (_globals.unk14 == 2) {
                    _globals.unk18 = 3;
                } else {
                    _globals.unkC = 4;
                }
block_10:
                var_eax += 1;
                if (var_eax >= arg0) {
                    _globals.unk10 = 5;
                    return;
                }
                goto loop_3;
            }
            _globals.unk14 = 3;
            /* Duplicate return node #14. Try simplifying control flow for better match */
            _globals.unk10 = 5;
            return;
        }
        _globals.unk8 = 3;
        _globals.unk10 = 5;
        return;
    }
    _globals.unk10 = 5;
}
