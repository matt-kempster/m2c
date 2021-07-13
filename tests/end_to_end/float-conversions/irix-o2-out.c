extern f32 D_410230;
extern f64 D_410238;
extern u32 D_410240;

void test(void) {
    f32 temp_f8;
    f64 temp_f18;
    u32 temp_t0;
    u32 temp_t1;
    f64 phi_f18;
    f32 phi_f8;

    D_410240 = (u32) D_410230;
    D_410240 = (u32) D_410238;
    temp_t0 = D_410240;
    temp_f18 = (f64) temp_t0;
    phi_f18 = temp_f18;
    if ((s32) temp_t0 < 0) {
        phi_f18 = temp_f18 + 4294967296.0;
    }
    D_410238 = phi_f18;
    temp_t1 = D_410240;
    temp_f8 = (f32) temp_t1;
    phi_f8 = temp_f8;
    if ((s32) temp_t1 < 0) {
        phi_f8 = temp_f8 + 4294967296.0f;
    }
    D_410230 = phi_f8;
}
