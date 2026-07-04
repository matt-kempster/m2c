enum my_enum _array[4] = { THREE, TWO_TOO, TWO_TOO, ZERO }; /* const */

enum my_enum test(enum my_enum x) {
    switch (x) {                                    /* irregular */
    default:
        return ZERO;
    case THREE:
        return _array[2];
    case TWO_TOO:
        return _array[1];
    case ZERO:
        return _array[0];
    }
}
