void test(s8 arg0); // static
extern ?32 D_410140;

void test(s8 arg0) {
    D_410140 = (?32) arg0;
    D_410140 = (?32) (s8) (arg0 * 2);
    D_410140 = (?32) (s8) (arg0 * 3);
    D_410140 = (?32) (s16) arg0;
    D_410140 = (?32) (s16) (arg0 * 2);
    D_410140 = (?32) (s16) (arg0 * 3);
}
