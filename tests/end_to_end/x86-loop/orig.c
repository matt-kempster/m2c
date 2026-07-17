/* Compiled with Microsoft Visual C++ 6.0 (CL.EXE) /O2. */
/* A global pointer forces the base-pointer reload on every iteration. */
int *arr_ptr;

void test(int n) {
    int i;
    for (i = 0; i < n; i++)
        arr_ptr[i] = i;
}
