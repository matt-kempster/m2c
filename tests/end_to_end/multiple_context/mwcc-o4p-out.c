f32 test(shape_t *s) {
    s32 temp_r0;

    temp_r0 = s->type;
    if (temp_r0 != 1) {
        if (temp_r0 < 1) {
            if (temp_r0 < 0) {
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
