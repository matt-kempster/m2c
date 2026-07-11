f64 test(f64 a, s32 b, f64 c) {
    f64 temp_f0;

    temp_f0 = (((f64) b * a) + (a / c)) - 7.0;
    if ((temp_f0 >= c) && (temp_f0 != c)) {
        if (!(temp_f0 > 9.0)) {
            global = 6.0;
            return 6.0;
        }
        /* Duplicate return node #5. Try simplifying control flow for better match */
        global = 5.0;
        return 5.0;
    }
    global = 5.0;
    return 5.0;
}
