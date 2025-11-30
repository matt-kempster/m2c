s32 __addsf3(f32, s32);                             /* extern */

f32 test(shape_t *s) {
    s32 temp_r0;

    temp_r0 = s->type;
    if ((temp_r0 != SHAPE_CIRCLE) && ((u32) temp_r0 >= 1U) && (temp_r0 != SHAPE_TRIANGLE)) {
        return 0.0f;
    }
    return (bitwise f32) __addsf3(s->origin.x, s->unkC);
}
