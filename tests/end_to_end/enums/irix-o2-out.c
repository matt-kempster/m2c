enum my_enum array[4] = { THREE, TWO_TOO, TWO_TOO, ZERO }; /* const */

enum my_enum test(enum my_enum x) {
    switch (x) {                                    /* irregular */
    case ZERO:
        return array[0];
    case TWO_TOO:
        return array[0];
    case THREE:
        return array[0];
    default:
        return ZERO;
    }
}
