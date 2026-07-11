enum my_enum array[4] = { THREE, TWO_TOO, TWO_TOO, ZERO }; /* const */

enum my_enum test(enum my_enum x) {
    switch (x) {                                    /* irregular */
    default:
        return ZERO;
    case THREE:
        return array[2];
    case TWO_TOO:
        return array[1];
    case ZERO:
        return array[0];
    }
}
