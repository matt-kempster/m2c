signed char test(signed char *ptr) {
    return ptr[3];
}

signed short test_loadw_disp(signed short *ptr) {
    return ptr[3];
}

signed test_loadl_disp(signed *ptr) {
    return ptr[3];
}

void test_storeb_disp(signed char *ptr, signed char value) {
    ptr[3] = value;
}

void test_storew_disp(signed short *ptr, signed short value) {
    ptr[3] = value;
}

void test_storel_disp(signed *ptr, signed value) {
    ptr[3] = value;
}

signed char test_loadb_indexed(signed char *ptr, unsigned index) {
    return ptr[index];
}

signed short test_loadw_indexed(signed short *ptr, unsigned index) {
    return ptr[index];
}

signed test_loadl_indexed(signed *ptr, unsigned index) {
    return ptr[index];
}

void test_storeb_indexed(signed char *ptr, unsigned index, signed char value) {
    ptr[index] = value;
}

void test_storew_indexed(signed short *ptr, unsigned index, signed short value) {
    ptr[index] = value;
}

void test_storel_indexed(signed *ptr, unsigned index, signed value) {
    ptr[index] = value;
}
