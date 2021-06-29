struct A {
    int x[5];
};

//void extern_fn(struct A *a);
//extern float extern_float;
//void static_fn(struct A *a) { }

static int static_int;
static struct A static_A = {{1,2,3,4,5}};
static struct A *static_A_ptr = &static_A;

static int static_array[3] = {2, 4, 6};
static const int static_ro_array[3] = {7, 8, 9};
static int static_bss_array[3];

int test(void) {
    static_int *= 456;
    extern_float *= 456.0f;
    static_fn(&static_A);
    extern_fn(static_A_ptr);
    static_bss_array[0] = static_array[0] + static_ro_array[0];
    return static_int;
}

/*
.rodata
glabel static_A
.word 0x01, 0x02, 0x03, 0x04, 0x05

glabel static_ro_array
.word 0x07, 0x08, 0x09

.data
glabel static_A_ptr
.word static_A

glabel static_array
.word 2, 4, 6

.bss
glabel static_int
.space 4

glabel static_bss_array
.space 12
*/
