struct B {
    int more_padding;
    int x;
};
struct A {
    int padding;
    int *array[10];
    struct B array2[10];
    int y;
};

struct NotExtended {
    int x;
    int y;
};
struct Extended {
    struct NotExtended x;
    int y;
    int z;
};
struct Outer {
    struct Extended x;
    struct NotExtended y;
    int z;
};

volatile int glob;

void Outer_Init(struct Extended *ext) {
    // mips_to_c should guess that `ext` is cast to `struct Outer *` based on the function name
    struct Outer *outer = (struct Outer *) ext;
    outer->x.x.x = 1;
    outer->x.x.y = 2;
    outer->x.y = 3;
    outer->x.z = 4;
    outer->y.x = 5;
    outer->y.y = 6;
    outer->z = 7;
}

void test(struct A *a, int b, struct Extended *ext, struct NotExtended *not, struct Outer *outer) {
    // Test array access inside structs
    glob = (int)a->array[b];
    glob = (int)&a->array[b];
    glob = a->array2[b].x;
    glob = (int)&a->array2[b].x;
    glob = a[b].y;
    glob = a->array2[3].x;
    glob = (int)&a->array2[3].x;

    // Perform normal, in-bounds struct access
    glob = ext->x.x;
    glob = ext->y;
    glob = not->x;
    glob = outer->x.x.x;
    glob = outer->y.x;
    glob = outer->z;

    // Perform array access
    glob = ext[-1].x.x;
    glob = ext[-1].y;
    glob = not[-1].x;
    glob = outer[-1].x.x.x;
    glob = outer[-1].y.x;
    glob = outer[-1].z;

    glob = ext[10].x.x;
    glob = ext[10].y;
    glob = not[10].x;
    glob = outer[10].x.x.x;
    glob = outer[10].y.x;
    glob = outer[10].z;

    // Use the Outer_Init function
    Outer_Init(ext);
}
