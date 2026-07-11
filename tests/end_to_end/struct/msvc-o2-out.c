Blah *test(Blah *b, Blah *b2) {
    s32 temp_ecx;

    temp_ecx = b->a;
    b->b += temp_ecx;
    b2->a = temp_ecx;
    b2->b = b->b;
    return b;
}
