// Test context with a union that has multiple fields of the same size
typedef int s32;
typedef float f32;

union TestUnion {
    f32 float_field;
    void *ptr_field;
    s32 int_field;
};

struct TestStruct {
    union TestUnion data;
};

s32 test(struct TestStruct *arg0, s32 arg1);
