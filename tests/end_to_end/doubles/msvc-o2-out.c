static f64 _real_4014000000000000 = 5.0;            /* const */
static f64 _real_4018000000000000 = 6.0;            /* const */
static f64 _real_4022000000000000 = 9.0;            /* const */
static f64 _real_401c000000000000 = 7.0;            /* const */

f64 test(f64 a, s32 b, f64 c) {
    f64 temp_f0;

    temp_f0 = (((f64) b * a) + (a / c)) - _real_401c000000000000;
    if ((temp_f0 >= c) && (temp_f0 != c)) {
        if (!(temp_f0 > _real_4022000000000000)) {
            _global = _real_4018000000000000;
            return _real_4018000000000000;
        }
        /* Duplicate return node #5. Try simplifying control flow for better match */
        _global = _real_4014000000000000;
        return _real_4014000000000000;
    }
    _global = _real_4014000000000000;
    return _real_4014000000000000;
}
