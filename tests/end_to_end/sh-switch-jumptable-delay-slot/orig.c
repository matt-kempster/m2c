extern int sink(int);

int test(int *values, int result) {
    int highest = 0;
    int i;

    for (i = 1; i < 4; i++) {
        if (values[i] > values[highest]) {
            highest = i;
        }
    }

    switch (highest) {
    case 0:
        result *= 2;
        break;
    case 1:
        result += 3;
        break;
    case 2:
        result += sink(result);
        break;
    case 3:
        result -= sink(result);
        break;
    }
    return result;
}
