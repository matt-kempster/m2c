? memcpy(? *, ? *, s32);                            /* extern */

s32 test(s32 arg0, s32 arg1, s32 arg2) {
    s32 temp_r2;

    memcpy(&unksp0, "hello", 6);
    temp_r2 = arg0 * 4;
    return (*(temp_r2 + arg1) * *(&unksp0 + arg0)) + *((arg0 * 2) + .L3.unk4) + *(arg2 + 4 + temp_r2);
}
