s32 test(s32 *arg0); // static

s32 test(s32 *arg0) {
    s32 temp_v0;

    temp_v0 = (s32) *arg0 % 2;
    *arg0 = temp_v0;
    return temp_v0;
}
