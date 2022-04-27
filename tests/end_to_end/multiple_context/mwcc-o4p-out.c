f32 test(shape_t *s) {
    s32 temp_r0;

    temp_r0 = s->type;
    if (temp_r0 != SHAPE_CIRCLE) {
        if (temp_r0 < SHAPE_CIRCLE) {
            if (temp_r0 < SHAPE_SQUARE) {
                return 0.0f;
            }
            return s->origin.x + s->unkC;
        }
        if (temp_r0 < 3) {
            return s->origin.x + s->unkC;
        }
        return 0.0f;
    }
    return s->origin.x + s->unkC;
}
