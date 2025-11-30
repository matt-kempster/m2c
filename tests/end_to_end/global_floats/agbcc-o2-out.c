s32 __addsf3(f32, f32);                             /* extern */
s32 __divsf3(s32, s32);                             /* extern */
s32 __mulsf3(s32, s32);                             /* extern */
f32 D_410150 = 1.23f;
f32 D_410154[3] = { 3.0f, 4.0f, 5.0f };
f32 D_410160[3] = { 6.0f, 7.0f, 8.0f };
f32 D_400120[3] = { 10.0f, 11.0f, 12.0f };          /* const */
f32 D_40012C[5] = { 14.0f, 15.0f, 16.0f, 17.0f, 18.0f }; /* const */

f32 test(s32 i) {
    s32 temp_r0;
    s32 temp_r0_2;

    temp_r0_2 = __addsf3(D_410160[i], D_400120[i]);
    D_410170[i] = (bitwise f32) temp_r0_2;
    temp_r0 = __mulsf3((bitwise s32) D_410150, __addsf3((bitwise f32) __mulsf3(temp_r0_2, 0x40B570A4), D_40012C[i]));
    D_410150 = (bitwise f32) temp_r0;
    return (bitwise f32) __divsf3(temp_r0, temp_r0_2);
}
