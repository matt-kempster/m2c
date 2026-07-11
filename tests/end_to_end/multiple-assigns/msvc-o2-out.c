s32 test(s32 x) {
    s32 temp_ecx;
    s32 temp_ecx_2;
    s32 temp_ecx_3;
    s32 temp_ecx_4;
    s32 temp_ecx_5;
    s32 var_ecx;

    var_ecx = x;
    if (var_ecx == 5) {
        do {
            glob = var_ecx;
            temp_ecx_2 = var_ecx + 1;
            glob = temp_ecx_2;
            temp_ecx_3 = temp_ecx_2 + 1;
            glob = temp_ecx_3;
            temp_ecx = temp_ecx_3 + 1;
            glob = temp_ecx;
            temp_ecx_4 = temp_ecx + 1;
            glob = temp_ecx_4;
            glob = temp_ecx_4;
            temp_ecx_5 = temp_ecx_4 + 1;
            glob = temp_ecx_5;
            var_ecx = temp_ecx_5 + 1;
            glob = temp_ecx;
        } while (var_ecx == 5);
        return temp_ecx;
    }
    return x;
}
