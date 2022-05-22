void test(s32 arg0, void *arg1) {
    s32 var_t1;
    void *var_a1;
    s32 phi_t2;

    var_a1 = arg1;
    var_t1 = 0;
    do {
        if (arg0 != 0) {
            phi_t2 = var_a1->unk1;
        } else {
            phi_t2 = var_a1->unk1;
        }
        var_t1 += phi_t2;
        var_a1 = NULL;
    } while (var_t1 != 0);
}
