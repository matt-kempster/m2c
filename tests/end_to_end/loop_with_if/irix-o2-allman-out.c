s32 test(s32 arg0)
{
    s32 phi_v1;

    phi_v1 = 0;
    if (arg0 > 0)
    {
        do
        {
            if (phi_v1 == 5)
            {
                phi_v1 *= 2;
            }
            else
            {
                phi_v1 += 4;
            }
        } while (phi_v1 < arg0);
    }
    return phi_v1;
}
