s32 test(u32 arg0, s32 arg1) {
    u32 temp_eax;

    temp_eax = arg0 + arg1;
    if ((temp_eax >= arg0) && (temp_eax != 1)) {
        return 0;
    }
    return 1;
}
