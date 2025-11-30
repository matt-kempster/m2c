extern s32 glob;

void test(s32 arg0) {
    if (arg0 >= 1) {
        if (arg0 <= 4) {
            glob = 1;
            if (arg0 == 1) {
                goto block_8;
            }
        } else if (arg0 <= 6) {
            if (arg0 == 1) {
                glob = arg0;
            } else {
block_8:
                glob = 2;
            }
        }
        if (arg0 >= 1) {
            if (arg0 <= 4) {
                glob = 1;
                if (arg0 == 1) {
                    goto block_17;
                }
            } else if (arg0 <= 6) {
                if (arg0 == 1) {
                    glob = arg0;
                    return;
                }
block_17:
                glob = 2;
            }
        }
    }
}
