static ? $$29;                                      /* unable to generate initializer: unknown type; const */

void test(Vec *b) {
        sp14 = $$29;                                    /* size 0xC */
        sp8 = sp14;                                     /* size 0xC */
        sp8 = *b;                                       /* size 0xC */
}

void test2(Vec *b) {
        sp8 = b->unk0;                                  /* size 0xC */
    b->z = 4.0f;
}
