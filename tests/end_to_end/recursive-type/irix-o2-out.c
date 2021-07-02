? func_00400090(? *, ? *); // static
void test(? *arg0, ? *arg1); // static

void test(? *arg0, ? *arg1) {
    ? *temp_a0;
    ? *temp_a1;

    temp_a1 = &arg1;
    temp_a0 = &arg0;
    arg0 = temp_a0;
    arg1 = temp_a1;
    func_00400090(temp_a0, temp_a1);
    arg0 = (? *) arg1;
    func_00400090((? *) arg1, arg1);
}
