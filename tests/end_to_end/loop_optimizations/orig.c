void test(char *foo, int length)
{
    int i;
    int total1, total2, total3;

    // array zeroing
    for (i = 0; i < length; i++)
    {
        foo[i] = 0;
    }

    // simple addition
    total1 = 0;
    for (i = 0; i < length; i++)
    {
        total1 += 1;
    }

    // simple multiplication
    total2 = 1;
    for (i = 1; i < length; i++)
    {
        total2 *= i;
    }

    // addition and multiplication
    total3 = 0;
    for (i = 0; i < length; i++)
    {
        total3 += i;
        total3 *= i;
    }
}
