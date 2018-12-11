f64 test(f32 arg0, s32 arg2, f64 arg4)
{
    f64 temp_f0;

    temp_f0 = ((((f64) arg2 * arg0) + (arg0 / arg4)) - 7.0);
    if (temp_f0 < arg4)
    {
        return 5.0;
    }
    if (temp_f0 == arg4)
    {
        return 5.0;
    }
    if (9.0 < temp_f0)
    {
        return 5.0;
    }
    return 6.0;
}
