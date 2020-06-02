void foo(signed** ptr) {}

void test(void) {
    signed* var;
    foo(&var);
}
