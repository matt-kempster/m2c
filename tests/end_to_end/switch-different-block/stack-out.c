extern s32 D_410170;

s32 test(s32 arg0) {
    s32 *temp_at;
    s32 var_a0;
    s32 var_a0_2;

    var_a0 = arg0;
    temp_at = &jtbl_400150 + ((var_a0 - 1) * 4);
    arg0 = *temp_at;
    if (temp_at != NULL) {
        switch (arg0) {                             /* unable to parse jump table */
        case 0:
            return var_a0 * var_a0;
        case 1:
            var_a0 -= 1;
            /* fallthrough */
        case 2:
            return var_a0 * 2;
        case 3:
            var_a0_2 = var_a0 + 1;
            /* Duplicate return node #8. Try simplifying control flow for better match */
            D_410170 = var_a0_2;
            return 2;
        case 5:
        case 6:
            var_a0_2 = var_a0 * 2;
            /* Duplicate return node #8. Try simplifying control flow for better match */
            D_410170 = var_a0_2;
            return 2;
        }
    } else {
    case 4:
        var_a0_2 = var_a0 / 2;
        D_410170 = var_a0_2;
        return 2;
    }
}
