extern s32 counter;
extern s8 flag;
static s32 table[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

s32 test(s32 arg0, s32 arg1) {
    s32 temp_eax;

    temp_eax = counter + 1;
    counter = temp_eax;
    flag = 1;
    table[arg0] += temp_eax;
    return *(arg1 + ((arg0 * 8) + 0x10));
}
