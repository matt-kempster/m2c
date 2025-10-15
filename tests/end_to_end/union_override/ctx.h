// Test context with a union that has multiple fields of the same size
typedef int s32;
typedef float f32;

union TestUnion {
    s32 int_field;
    f32 float_field;
    void *ptr_field;
};

struct TestStruct {
    union TestUnion data;
};
