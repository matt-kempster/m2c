f32 mix(f32, f32);                                  /* extern */

f32 test(f32 arg0, f32 arg1) {
    return mix(arg0 * arg1, arg0 - arg1) + arg0;
}
