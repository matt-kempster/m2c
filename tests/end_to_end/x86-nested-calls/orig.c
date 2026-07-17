/* Compiled with Microsoft Visual C++ 6.0 (CL.EXE) /O2. */
int inner(int);
int outer(int, int);

int test(int a, int b) {
    return outer(inner(a), b);
}
