void test(s32 x) {
    u32 temp_eax;

    temp_eax = x - 1;
    if (temp_eax <= 5U) {
        switch (x) {                                /* switch 1 */
        case 1:                                     /* switch 1 */
        case 2:                                     /* switch 1 */
        case 3:                                     /* switch 1 */
        case 4:                                     /* switch 1 */
            _glob = 1;
            if (x == 1) {
                _glob = 2;
            }
            break;
        case 5:                                     /* switch 1 */
        case 6:                                     /* switch 1 */
            _glob = (u8) (x != 1) + 1;
            break;
        }
        if (temp_eax <= 5U) {
            switch (x) {                            /* switch 2 */
            case 1:                                 /* switch 2 */
            case 2:                                 /* switch 2 */
            case 3:                                 /* switch 2 */
            case 4:                                 /* switch 2 */
                _glob = 1;
                if (x == 1) {
                    _glob = 2;
                    return;
                }
                break;
            case 5:                                 /* switch 2 */
            case 6:                                 /* switch 2 */
                _glob = (u8) (x != 1) + 1;
                break;
            }
        }
    }
}
