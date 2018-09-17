f32 test(f32 f12, s32 a1, s32 a3) {
    D_410140 = (s32) f12;
    D_410144 = (f32) a1;
    if ((a3 + 3) < 0)
    {
        return;
        // (possible return value: ((f32) (a3 + 3) + (f32) ((f64) (f32) (f64) (f32) ((f64) a2 + 5.0) + D_400130)))
    }
    // (possible return value: ((f32) (a3 + 3) + (f32) ((f64) (f32) (f64) (f32) ((f64) a2 + 5.0) + D_400130)))
}
