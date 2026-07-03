void test(void *arg0, s32 arg1, s32 arg2) {
    arg0->unkA = (s16) arg1;
    arg0->unk8 = 2;
    arg0->unkC = (s16) arg2;
    arg0->unkE = (s16) (arg1 + arg2);
}
