Warning: confusing control flow, output may have incorrect && and || detection. Run with --no-andor to disable detection and print gotos instead.

f64 test(f32 arg0, s32 arg2, f32 arg4, f32 arg5)
{
    f64 temp_f0;
    f32 phi_f3;

    temp_f0 = (((f64) arg2 * arg0) + (arg0 / arg5)) - 7.0;
    if (!(temp_f0 < arg5))
    {
        if ((temp_f0 == arg5) || (9.0 < temp_f0))
        {
block_4:
            phi_f3 = 2.3125f;
        }
        else
        {
            phi_f3 = 2.375f;
        }
    }
    else
    {
        goto block_4;
    }
    D_410150 = (f32) phi_f3;
    D_410150 = (f32) 0.0f;
    return (f64) 0.0f;
}
