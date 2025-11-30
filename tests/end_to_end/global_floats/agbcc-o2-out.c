s32 __addsf3(f32, f32);                             /* extern */
s32 __divsf3(s32, s32);                             /* extern */
s32 __mulsf3(s32, s32);                             /* extern */
f32 D_410150 = -5.2174897e-17f;
f32 D_410154[3] = { 2.3049e-41f, 4.6007e-41f, 5.7487e-41f };
f32 D_410160[3] = { 6.8966e-41f, 8.0446e-41f, 9.1e-44f };
f32 D_400120[3] = { 1.157e-41f, 1.731e-41f, 2.305e-41f }; /* const */
f32 D_40012C[5] = { 3.453e-41f, 4.0269e-41f, 4.6009e-41f, 4.8879e-41f, 5.1749e-41f }; /* const */

f32 test(s32 i) {
    s32 temp_r0;
    s32 temp_r0_2;

    temp_r0_2 = __addsf3(D_410160[i], D_400120[i]);
    D_410170[i] = (bitwise f32) temp_r0_2;
    temp_r0 = __mulsf3((bitwise s32) D_410150, __addsf3((bitwise f32) __mulsf3(temp_r0_2, 0x40B570A4), D_40012C[i]));
    D_410150 = (bitwise f32) temp_r0;
    return (bitwise f32) __divsf3(temp_r0, temp_r0_2);
}
