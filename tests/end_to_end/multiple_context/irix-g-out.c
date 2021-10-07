f32 test(shape_t *s) {
    shape_t *spC;
    shape_t *sp8;
    shape_t *sp4;
    s32 temp_a1;

    temp_a1 = s->type;
    if (temp_a1 != 0) {
        if (temp_a1 != 1) {
            if (temp_a1 != 2) {
                return 0.0f;
            }
            sp4 = s;
            return sp4->origin.x + sp4->unkC;
        }
        sp8 = s;
        return sp8->origin.x + sp8->unkC;
    }
    spC = s;
    return spC->origin.x + spC->unkC;
}
