s32 __addsf3(s32, s32);                             /* extern */
extern s32 globalf;
extern s32 globali;

void test(s32 arg0, s32 arg1, s32 arg2, s32 arg3, s32 arg4, s32 arg5) {
    globalf = __addsf3(__addsf3(arg0, arg2), arg4);
    globali = arg1 + arg3 + arg5;
}
