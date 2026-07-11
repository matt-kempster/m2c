static f32 real_00000000 = 0.0f;                    /* const */

f32 test(shape_t *s) {
    s32 temp_eax;

    temp_eax = s->type;
    if ((temp_eax != SHAPE_SQUARE) && (temp_eax != SHAPE_CIRCLE) && (temp_eax != SHAPE_TRIANGLE)) {
        return real_00000000;
    }
    return s->unkC + s->origin.x;
}
