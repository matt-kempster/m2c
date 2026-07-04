s32 read(s32, s8 *, ?);                             /* static */
s32 write(s32, s8 *, s32);                          /* static */

s32 test(s32 in, s32 out) {
    s8 spC;
    s32 var_eax;

    var_eax = read(in, &spC, 0x123456);
    if (var_eax >= 0) {
        spC = spC ^ 0x55;
        *(sp + (var_eax + 0xB)) = (s8) *(sp + (var_eax + 0xB)) ^ 0x55;
        var_eax = write(out, &spC, var_eax);
    }
    return var_eax;
}
