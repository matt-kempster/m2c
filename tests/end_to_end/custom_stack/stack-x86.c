struct Vec {
    int x;
    int y;
    int z;
};

void frob(void *x) {}

int test(struct Vec *v) {
    char a;
    short b;
    int c;
    struct Vec *d;
    struct Vec e;

    frob(&a);
    frob(&b);
    frob(&c);
    frob(&d);
    frob(&e);

    a = v->x + v->y;
    b = v->x + v->z;
    c = v->y + v->z;

    e.x = v->x * a;
    e.y = v->y * b;
    e.z = v->z * c;

    if (a) {
        d = v;
    } else {
        d = &e;
    }

    return a + b + c + d->x + e.y;
}

struct _m2c_stack_test {
    char pad0[39];
    char a;
    struct Vec *d;
    short b;
    char pad1[2];
    int c;
    struct Vec e;
};
