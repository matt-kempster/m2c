Large test(Large *__return__, s32 value) {
    s32 sp0;
    s32 sp4;
    s32 sp8;
    s32 temp_r1;
    s32 temp_r1_2;

    sp0 = value;
    temp_r1 = value + 1;
    sp4 = temp_r1;
    temp_r1_2 = value + 2;
    sp8 = temp_r1_2;
    __return__->a = value;
    __return__->b = temp_r1;
    __return__->c = temp_r1_2;
    return (Large) __return__;
}
