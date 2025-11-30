s32 test(struct A *a, struct C *c, s32 i) {
    return a->ys[i] + *(&c->bs[0].y + (i * 8));
}
