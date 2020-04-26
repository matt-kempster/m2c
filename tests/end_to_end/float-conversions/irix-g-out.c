void test(void)
{
    f64 temp_f18;
    f64 phi_f18;

    D_410260 = (u32) D_410250;
    D_410260 = (u32) D_410258;
    temp_f18 = (f64) D_410260;
    phi_f18 = temp_f18;
    if (D_410260 < 0)
    {
        phi_f18 = temp_f18 + 4294967296.0;
    }
    D_410258 = phi_f18;
    D_410250 = (f32) (u32) D_410260;
}
