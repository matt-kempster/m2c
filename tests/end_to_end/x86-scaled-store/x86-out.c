extern ? table;

void test(s32 arg0, s32 arg1) {
    *((arg1 * 4) + &table) = arg0;
    *(arg1 + ((arg0 * 2) + 1)) = 7;
}
