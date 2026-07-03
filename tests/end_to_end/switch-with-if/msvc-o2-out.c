extern s32 _glob;

void test(s32 arg0) {
    u32 temp_eax;

    temp_eax = arg0 - 1;
    if (temp_eax <= 5U) {
        switch (arg0) {                             /* switch 1 */
        case 1:                                     /* switch 1 */
        case 2:                                     /* switch 1 */
        case 3:                                     /* switch 1 */
        case 4:                                     /* switch 1 */
            _glob = 1;
            if (arg0 == 1) {
                _glob = 2;
            }
            break;
        case 5:                                     /* switch 1 */
        case 6:                                     /* switch 1 */
            _glob = (u8) (arg0 != 1) + 1;
            break;
        }
        if (temp_eax <= 5U) {
            switch (arg0) {                         /* switch 2 */
            case 1:                                 /* switch 2 */
            case 2:                                 /* switch 2 */
            case 3:                                 /* switch 2 */
            case 4:                                 /* switch 2 */
                _glob = 1;
                if (arg0 == 1) {
                    _glob = 2;
                    return;
                }
                break;
            case 5:                                 /* switch 2 */
            case 6:                                 /* switch 2 */
                _glob = (u8) (arg0 != 1) + 1;
                break;
            }
        }
    }
}
