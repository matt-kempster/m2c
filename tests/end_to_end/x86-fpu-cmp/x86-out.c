s32 test(f32 arg0, f32 arg1, f32 arg2) {
    if ((arg0 > arg1) && !(arg0 >= arg2)) {
        return 1;
    }
    return 0;
}
