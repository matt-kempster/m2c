struct A {
    int x[5];
};

//void extern_fn(struct A *a);
//extern float extern_float;
//void static_fn(struct A *a) { }

static int static_int;
static struct A static_A = {{1,2,3,4,5}};
static struct A *static_A_ptr = &static_A;

int test(void) {
    static_int *= 456;
    extern_float *= 456.0f;
    static_fn(&static_A);
    extern_fn(static_A_ptr);
    return static_int;
}

