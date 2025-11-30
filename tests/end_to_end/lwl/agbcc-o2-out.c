? memcpy(? *, ? *, s32);                            /* extern */
? foo(? *);                                         /* static */

void test(void) {
    memcpy(&unksp0, "abcdef", 7);
    foo(&unksp0);
    .L4.unk4->unk4 = (s32) .L4.unk8->unk4;
    .L4.unkC->unk0 = (s32) .L4.unk4->unk0;
    .L4.unkC->unk4 = (s32) .L4.unk4->unk4;
    memcpy(.L4.unk10, .L4.unk14, 4);
}
