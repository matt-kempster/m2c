int test(int (*callback)(int), int value) {
    return callback(value) + 1;
}
