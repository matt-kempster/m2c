extern s32 arr_ptr;

void test(s32 arg0) {
    s32 var_eax;

    var_eax = 0;
    if (arg0 > 0) {
        do {
            *(arr_ptr + (var_eax * 4)) = var_eax;
            var_eax += 1;
        } while (var_eax < arg0);
    }
}
