s32 test(void *arg0) {
    s32 temp_v0;

    temp_v0 = *arg0;
    *arg0 = (s32) (temp_v0 % 2);
    return temp_v0;
}
