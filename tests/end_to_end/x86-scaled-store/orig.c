/* Compiled with Microsoft Visual C++ 6.0 (CL.EXE) /O1. */
int table[256];

void test(int val, int idx) {
    table[idx] = val;
    *(char *)(idx + val * 2 + 1) = 7;
}
