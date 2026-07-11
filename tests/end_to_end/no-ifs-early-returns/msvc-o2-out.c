s32 test(struct A *a, struct A *b) {
    s32 temp_eax;

    temp_eax = a->x;
    if (temp_eax != 8) {
        if (temp_eax == 0xF) {
            b->x -= 0xF;
            return 0;
        }
        /* Duplicate return node #4. Try simplifying control flow for better match */
        return 0;
    }
    b->x += 8;
    return 0;
}
