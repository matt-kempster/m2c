? memcpy(? *, ? *, s32);                            /* extern */
? foo(? *);                                         /* static */
extern ? a1;
extern ? a2;
extern ? a3;
extern ? buf;

void test(void) {
    memcpy(&unksp0, "abcdef", 7);
    foo(&unksp0);
    a1.unk4 = (s32) a2.unk4;
    a3.unk0 = (s32) a1.unk0;
    a3.unk4 = (s32) a1.unk4;
    memcpy(&buf, "ghi", 4);
}
