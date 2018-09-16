test(f32 f12, s32 a1, s32 a3) {
    (s32) D_410120 = (s32) f12;
    (f32) D_410124 = (f32) a1;
    if (((a3 + 3) < 0))
    {
        return;
        // (possible return value: ((f32) (a3 + 3) + (f32) ((f64) a2 + 0.0f)))
    }
    // (possible return value: ((f32) (a3 + 3) + (f32) ((f64) a2 + 0.0f)))
}
