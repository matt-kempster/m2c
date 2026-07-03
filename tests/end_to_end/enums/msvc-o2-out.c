enum my_enum _array[4] = { THREE, TWO_TOO, TWO_TOO, ZERO }; /* const */

enum my_enum test(enum my_enum x) {
    if (x != ZERO) {
        if (x != TWO_TOO) {
            if ((x - 2) != 1) {
                return ZERO;
            }
            return _array[2];
        }
        return _array[1];
    }
    return _array[0];
}
