s16 test(struct SomeStruct *arg, u8 should, ? union_arg, ? union_arg_unk4, ...)
{
    s8 temp_t6;

    temp_t6 = should & 0xFF;
    if (temp_t6 != 0)
    {
        globalf = (f32) arg->float_field;
        globali = (s32) arg->int_field;
        arg->data_field.double_innerfield = temp_t6;
    }
    else
    {
        arg->pointer_field = NULL;
        arg->data_field.double_innerfield = 0.0;
    }
    return arg->unk2;
}
