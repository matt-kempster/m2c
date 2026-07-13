// MSVC /O2 lowers the central range to two tables: a byte map from input
// values to unique cases, followed by a pointer table for those cases.
// The outlying cases are handled with separate comparisons.
int glob;

int test(int x) {
    switch (x) {
    case 1:
        return x*x;
    case 2:
        x--;
        // fallthrough
    case 3:
    case 101:
    case 200: // if check
        return x ^ (x+1);
    case 107:
        x++;
        break;
    case -50: // if check
        x--;
        break;
    case 50: // if check
        x++;
        break;
    case 6:
    case 7:
        x *= 2;
        // fallthrough

    case 102:
        if (glob) {
            break;
        }
        // fallthrough
    case 103:
    case 104:
    case 105:
    case 106:
        x--;
        // fallthrough

    default:
        x /= 2;
        break;
    }
    glob = x;
    return 2;
}
