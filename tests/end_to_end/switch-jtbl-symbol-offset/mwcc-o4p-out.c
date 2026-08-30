? fn_800EFFD4(s32);                                 /* extern */
extern void *__GXData;

void test(u32 arg0, s32 arg1, u32 arg2, s32 arg3, s32 arg4, s32 arg5) {
    s32 temp_r10;
    s32 var_r10;
    s32 var_r10_2;
    s32 var_r11;
    s32 var_r12;

    var_r10 = 0;
    var_r12 = 0;
    var_r11 = 5;
    switch (arg2) {                                 /* switch 1 */
    case 0:                                         /* switch 1 */
        var_r11 = 0;
        var_r12 = 1;
        break;
    case 1:                                         /* switch 1 */
        var_r11 = 1;
        var_r12 = 1;
        break;
    case 2:                                         /* switch 1 */
        var_r11 = 3;
        var_r12 = 1;
        break;
    case 3:                                         /* switch 1 */
        var_r11 = 4;
        var_r12 = 1;
        break;
    case 19:                                        /* switch 1 */
        var_r11 = 2;
        break;
    case 20:                                        /* switch 1 */
        var_r11 = 2;
        break;
    case 4:                                         /* switch 1 */
        var_r11 = 5;
        break;
    case 5:                                         /* switch 1 */
        var_r11 = 6;
        break;
    case 6:                                         /* switch 1 */
        var_r11 = 7;
        break;
    case 7:                                         /* switch 1 */
        var_r11 = 8;
        break;
    case 8:                                         /* switch 1 */
        var_r11 = 9;
        break;
    case 9:                                         /* switch 1 */
        var_r11 = 0xA;
        break;
    case 10:                                        /* switch 1 */
        var_r11 = 0xB;
        break;
    case 11:                                        /* switch 1 */
        var_r11 = 0xC;
        break;
    }
    switch (arg1) {                                 /* switch 2; irregular */
    case 1:                                         /* switch 2 */
        var_r10 = (((0 & ~2 & ~4) | ((var_r12 << 2) & 4)) & ~0x70 & ~0xF80) | ((var_r11 << 7) & 0xF80);
        break;
    case 0:                                         /* switch 2 */
        var_r10 = ((((0 | 2) & ~4) | ((var_r12 << 2) & 4)) & ~0x70 & ~0xF80) | ((var_r11 << 7) & 0xF80);
        break;
    default:                                        /* switch 2 */
        var_r10 = (((((((((0 & ~2 & ~4) | ((var_r12 << 2) & 4)) & ~0x70) | 0x10) & ~0xF80) | ((var_r11 << 7) & 0xF80)) & ~0x7000) | (((arg2 - 0xC) << 0xC) & 0x7000)) & ~0x38000) | (((arg1 - 2) << 0xF) & 0x38000);
        break;
    case 10:                                        /* switch 2 */
        temp_r10 = (0 & ~2 & ~4) | ((var_r12 << 2) & 4);
        if ((s32) arg2 == 0x13) {
            var_r10_2 = (temp_r10 & ~0x70) | 0x20;
        } else {
            var_r10_2 = (temp_r10 & ~0x70) | 0x30;
        }
        var_r10 = (var_r10_2 & ~0xF80) | 0x100;
        break;
    }
    *(s8 *)0xCC008000 = 0x10;
    *(s8 *)0xCC008000 = (s32) (arg0 + 0x1040);
    *(s8 *)0xCC008000 = var_r10;
    *(s8 *)0xCC008000 = 0x10;
    *(s8 *)0xCC008000 = (s32) (arg0 + 0x1050);
    *(s8 *)0xCC008000 = (s32) ((((0 & ~0x3F) | ((arg5 - 0x40) & 0x3F)) & ~0x100) | ((arg4 << 8) & 0x100));
    switch (arg0) {                                 /* switch 3 */
    case 0:                                         /* switch 3 */
        __GXData->unk80 = (s32) ((__GXData->unk80 & ~0xFC0) | ((arg3 << 6) & 0xFC0));
        break;
    case 1:                                         /* switch 3 */
        __GXData->unk80 = (s32) ((__GXData->unk80 & ~0x3F000) | ((arg3 << 0xC) & 0x3F000));
        break;
    case 2:                                         /* switch 3 */
        __GXData->unk80 = (s32) ((__GXData->unk80 & ~0xFC0000) | ((arg3 << 0x12) & 0xFC0000));
        break;
    case 3:                                         /* switch 3 */
        __GXData->unk80 = (s32) ((__GXData->unk80 & ~0x3F000000) | ((arg3 << 0x18) & 0x3F000000));
        break;
    case 4:                                         /* switch 3 */
        __GXData->unk84 = (s32) ((__GXData->unk84 & ~0x3F) | (arg3 & 0x3F));
        break;
    case 5:                                         /* switch 3 */
        __GXData->unk84 = (s32) ((__GXData->unk84 & ~0xFC0) | ((arg3 << 6) & 0xFC0));
        break;
    case 6:                                         /* switch 3 */
        __GXData->unk84 = (s32) ((__GXData->unk84 & ~0x3F000) | ((arg3 << 0xC) & 0x3F000));
        break;
    default:                                        /* switch 3 */
        __GXData->unk84 = (s32) ((__GXData->unk84 & ~0xFC0000) | ((arg3 << 0x12) & 0xFC0000));
        break;
    }
    fn_800EFFD4(arg0 + 1);
}
