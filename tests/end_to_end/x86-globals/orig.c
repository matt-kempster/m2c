/* Compiled with Microsoft Visual C++ 6.0 (CL.EXE) /O2. */
/* Volatile prevents CL.EXE from folding the separate table load/add/store. */
volatile int counter;
volatile signed char flag;
volatile int table[8];

int test(int idx, char *arg1) {
    int t;
    t = counter + 1;
    counter = t;
    flag = 1;
    table[idx] += t;
    return *(int *)(arg1 + (idx * 8) + 0x10);
}
