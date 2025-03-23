extern f32 myfloat;
extern s32 myint;

f32 test(s32 arg0) {
    f32 temp_f1;
    f32 var_f1;

    var_f1 = -myfloat;
    if (arg0 != 7) {
        temp_f1 = var_f1;
        var_f1 = 2.0f * myfloat;
        myint = (s32) temp_f1;
    }
    return var_f1;
}
