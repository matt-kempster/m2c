extern signed callee(signed value);

signed test(signed value) {
    if (value == 0) {
        return 1;
    }
    return 2;
}

signed test_if_nonzero(signed value) {
    if (value != 0) {
        return 3;
    }
    return 4;
}

signed test_if_greater(signed lhs, signed rhs) {
    if (lhs > rhs) {
        return lhs;
    }
    return rhs;
}

signed test_loop(signed value) {
    signed result = 0;
    while (value != 0) {
        result += value;
        value--;
    }
    return result;
}

signed test_call(signed value) {
    return callee(value) + 1;
}
