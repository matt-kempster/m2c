void *test(void)
{
    f32 temp_f8;
    f64 temp_f18;
    f64 phi_f18;
    f32 phi_f8;

    D_410240 = (u32) D_410230;
    D_410240 = (u32) D_410238;
    temp_f18 = (f64) D_410240;
    phi_f18 = temp_f18;
    if (D_410240 < 0)
    {
        phi_f18 = temp_f18 + 4294967296.0;
    }
    D_410238 = phi_f18;
    temp_f8 = (f32) D_410240;
    phi_f8 = temp_f8;
    if (D_410240 < 0)
    {
        phi_f8 = temp_f8 + 4294967296.0f;
    }
    D_410230 = phi_f8;
    return &D_410240;
}
