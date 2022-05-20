typedef struct {
    /* 0x0000 */ char pad0[0xC];
    /* 0x000C */ SomeStruct unkC;                   /* inferred */
    /* 0x0010 */ s32 whatever[0x1000];              /* overlap */
} SomeStruct;                                       /* size = 0x401C */

? func_800E2768(s16);                               /* extern */

void test(void) {
    SomeStruct *var_s0;
    s32 var_s1;

    var_s1 = 0;
    var_s0 = &glob;
    do {
        if (*NULL == 0) {
            func_800E2768(var_s0->unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unkC.unk0);
        }
        var_s1 += 1;
        var_s0 = &var_s0->unkC;
    } while (var_s1 < 5);
}
