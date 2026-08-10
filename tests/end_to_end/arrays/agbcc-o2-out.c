? memcpy(? *, ? *, s32);                            /* extern */
extern ? globalArray;

s32 test(s32 arg0, s32 arg1, s32 arg2) {
    ? sp0;
    s32 temp_r2;

    memcpy(&sp0, "hello", 6);
    temp_r2 = arg0 * 4;
    return (*(temp_r2 + arg1) * *(&sp0 + arg0)) + *((arg0 * 2) + &globalArray) + *(arg2 + 4 + temp_r2);
}
