void test(void) {
    no_args_func();
    func_with_args(2.0f, M2C_ERROR(/* Read from unset register $f14 */), M2C_ERROR(/* Read from unset register $a2 */), M2C_ERROR(/* Read from unset register $a3 */));
}
