static f64 real_4014000000000000 = 5.0;             /* const */
static f64 real_4018000000000000 = 6.0;             /* const */
static f64 real_4022000000000000 = 9.0;             /* const */
static f64 real_401c000000000000 = 7.0;             /* const */

f64 test(f64 a, s32 b, f64 c) {
    f64 temp_f0;

    temp_f0 = (((f64) b * a) + (a / c)) - real_401c000000000000;
    if ((temp_f0 >= c) && (temp_f0 != c)) {
        if (!(temp_f0 > real_4022000000000000)) {
            global = real_4018000000000000;
            return real_4018000000000000;
        }
        /* Duplicate return node #5. Try simplifying control flow for better match */
        global = real_4014000000000000;
        return real_4014000000000000;
    }
    global = real_4014000000000000;
    return real_4014000000000000;
}
