f32 test(shape_t *s) {
    s32 temp_v0;

    temp_v0 = s->type;
    if (temp_v0 != 0) {
        if (temp_v0 != 1) {
            if (temp_v0 != 2) {
                return 0.0f;
            }
            return s->origin.x + s->unkC;
        }
        return s->origin.x + s->unkC;
    }
    return s->origin.x + s->unkC;
}
