f32 test(shape_t *s) {
    s32 temp_v0;

    temp_v0 = s->type;
    switch (temp_v0) {                              /* irregular */
    case 0:
        return s->origin.x + s->unkC;
    case 1:
        return s->origin.x + s->unkC;
    case 2:
        return s->origin.x + s->unkC;
    default:
        return 0.0f;
    }
}
