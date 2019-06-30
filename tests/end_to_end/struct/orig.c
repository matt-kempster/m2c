struct Blah {
    int a, b;
};

struct Blah *test(struct Blah *b) {
    int c = b->a + b->b;
    b->b = c;
    return b;
}
