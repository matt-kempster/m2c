extern s32 _counter;
extern s8 _flag;
static s32 _table[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

s32 test(s32 arg0, s32 arg1) {
    s32 temp_eax;

    temp_eax = _counter + 1;
    _counter = temp_eax;
    _flag = 1;
    _table[arg0] += temp_eax;
    return *(arg1 + ((arg0 * 8) + 0x10));
}
