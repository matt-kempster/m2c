f32 test(f32 arg0, s32 arg1, f32 arg2, s32 arg3) {
    D_410140 = (s32) arg0;
    D_410144 = (f32) arg1;
    if ((arg3 + 3) < 0)
    {
        return;
        // (possible return value: ((f32) (arg3 + 3) + (f32) ((f64) (f32) (f64) (f32) ((f64) arg2 + 5.0) + D_400130)))
    }
    // (possible return value: ((f32) (arg3 + 3) + (f32) ((f64) (f32) (f64) (f32) ((f64) arg2 + 5.0) + D_400130)))
}
