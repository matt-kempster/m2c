f32 test(shape_t *s) {
    s32 temp_eax;

    temp_eax = s->type;
    if ((temp_eax != SHAPE_SQUARE) && (temp_eax != SHAPE_CIRCLE) && (temp_eax != SHAPE_TRIANGLE)) {
        return 0.0f;
    }
    return s->unkC + s->origin.x;
}
