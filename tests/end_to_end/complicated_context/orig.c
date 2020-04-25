int globali;
float globalf;

union SomeUnion {
    double double_innerfield;
    char char_innerfield;
};

enum SomeEnum
{
    FIRST_ELEM = 1,
    SECOND_ELEM = 2
};

struct SomeBitfield {
    char char_bit : 1;
    int int_bit : 4;
    short short_bit : 16;
    char : 0;
    unsigned char unsigned_bit : 7;
};

struct SomeStruct
{
    int int_field;
    float float_field;
    void *pointer_field;
    union SomeUnion data_field;
    enum SomeEnum enum_field;
    long long long_long_field;
    struct SomeBitfield bitfield_field;
    int array_arithmetic_1[1 + 1];
    int array_arithmetic_2[2 - 1];
    int array_arithmetic_3[1 * 1];
    int array_arithmetic_4[1 << 1];
    int array_arithmetic_5[1 >> 1];
};

void func_decl(void) {
    globali = 0;
}

short test(struct SomeStruct *arg, unsigned char should, union SomeUnion union_arg, ...)
{
    // This comment should be stripped
    /* This comment should also be stripped */
    /**
     * Even multi-line comments.
     */
    union SomeUnion stack_union;
    stack_union.double_innerfield = union_arg.double_innerfield;

    if (should)
    {
        globalf = arg->float_field;
        globali = arg->int_field;
        arg->data_field.char_innerfield = should;
    }
    else
    {
        arg->pointer_field = (void *)0;
        arg->data_field.double_innerfield = 0.;
    }
    return (short)arg->int_field;
}
