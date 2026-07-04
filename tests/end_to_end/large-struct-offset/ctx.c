struct A {
    char a[0x12345];
    int b;
};

extern int glob;

int *test(struct A *a);
