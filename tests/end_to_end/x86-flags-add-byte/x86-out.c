s32 test(u8 arg0, s8 arg1) {
    u8 temp_eax;

    temp_eax = arg0 + arg1;
    if (!((temp_eax >= arg0) && (temp_eax != 0))) {
        return 0;
    }
    return 1;
}
