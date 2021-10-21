struct _mips2c_stack_test {
    s8 pad0[32];                                    /* +0x0 */
    struct Vec e;                                   /* +0x20 */
    struct Vec *d;                                  /* +0x2C */
    s32 c;                                          /* +0x30 */
    s16 b;                                          /* +0x34 */
    s8 pad1[1];                                     /* +0x36 */
    s8 a;                                           /* +0x37 */
};                                                  /* size 0x38 */

? func_00400090(s8 *);                              /* static */

s32 test(struct Vec *v) {
    s8 a;
    s16 b;
    s32 c;
    struct Vec *d;
    struct Vec e;
    s32 temp_t4;

    func_00400090(&a);
    func_00400090((s8 *) &b);
    func_00400090((s8 *) &c);
    func_00400090((s8 *) &d);
    func_00400090((s8 *) &e);
    a = v->x + v->y;
    b = v->x + v->z;
    temp_t4 = v->y + v->z;
    c = temp_t4;
    e = v->x * a;
    e.y = v->y * b;
    e.z = v->z * temp_t4;
    if (a != 0) {
        d = v;
    } else {
        d = &e;
    }
    return a + b + c + d->x + e.y;
}
