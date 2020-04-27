void test(struct A *a, s32 b)
{
    D_410100 = (int *) a->array[b];
    D_410100 = (int *) &a->array[b];
    D_410100 = (int *) a->array2[b].x;
    D_410100 = &a->array2[b].x;
    D_410100 = (int *) a[b].y;
}
