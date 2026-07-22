signed char test(signed char *ptr) {
    return *ptr;
}

signed short test_loadw(signed short *ptr) {
    return *ptr;
}

signed test_loadl(signed *ptr) {
    return *ptr;
}

void test_storeb(signed char *ptr, signed char value) {
    *ptr = value;
}

void test_storew(signed short *ptr, signed short value) {
    *ptr = value;
}

void test_storel(signed *ptr, signed value) {
    *ptr = value;
}

signed char test_loadb_postinc(signed char **ptr) {
    signed char *current = *ptr;
    signed char value = *current++;
    *ptr = current;
    return value;
}

signed short test_loadw_postinc(signed short **ptr) {
    signed short *current = *ptr;
    signed short value = *current++;
    *ptr = current;
    return value;
}

signed test_loadl_postinc(signed **ptr) {
    signed *current = *ptr;
    signed value = *current++;
    *ptr = current;
    return value;
}

void test_storeb_predec(signed char **ptr, signed char value) {
    signed char *current = *ptr;
    *--current = value;
    *ptr = current;
}

void test_storew_predec(signed short **ptr, signed short value) {
    signed short *current = *ptr;
    *--current = value;
    *ptr = current;
}

void test_storel_predec(signed **ptr, signed value) {
    signed *current = *ptr;
    *--current = value;
    *ptr = current;
}
