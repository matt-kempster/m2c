/* Compiled with Microsoft Visual C++ 6.0 (CL.EXE) /O1. */
typedef unsigned int u32;
typedef unsigned __int64 u64;

int test(int a, int b, int c) {
    int t2, q, r;
    u32 hi;
    t2 = (a + b) - (c * 0x8c);
    q = t2 / b;
    r = -(t2 % b);
    hi = (u32)(((u64)(u32)r * (u32)b) >> 32);
    return q + (int)hi;
}
