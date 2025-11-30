s32 read(?, ? *, s32);                              /* static */
s32 write(s32, ? *, s32);                           /* static */

s32 test(? arg0, s32 arg1) {
    s32 temp_r0;
    u8 *temp_r1;

    temp_r0 = read(arg0, &unksp0, 0x123456);
    if (temp_r0 >= 0) {
        unksp0 = (u8) (unksp0 ^ 0x55);
        temp_r1 = &unksp0 + (temp_r0 - 1);
        *temp_r1 ^= 0x55;
        return write(arg1, &unksp0, temp_r0);
    }
    return temp_r0;
}
