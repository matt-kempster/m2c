? func_004000B0(struct Extended *);                 /* extern */
extern s32 *D_410200;

void test(struct A *a, s32 b, struct Extended *ext, struct NotExtended *not, struct Outer *outer) {
    D_410200 = a->array[b];
    D_410200 = (s32 *) &a->array[b];
    D_410200 = (s32 *) a->array2[b].x;
    D_410200 = &a->array2[b].x;
    D_410200 = (s32 *) a[b].y;
    D_410200 = (s32 *) a->array2[3].x;
    D_410200 = &a->array2[3].x;
    D_410200 = (s32 *) ext->x.x;
    D_410200 = (s32 *) ext->y;
    D_410200 = (s32 *) not->x;
    D_410200 = (s32 *) outer->x.x.x;
    D_410200 = (s32 *) outer->y.x;
    D_410200 = (s32 *) outer->z;
    D_410200 = ext->unk-10;
    D_410200 = ext->unk-8;
    D_410200 = (s32 *) not[-1].x;
    D_410200 = (s32 *) outer[-1].x.x.x;
    D_410200 = (s32 *) outer[-1].y.x;
    D_410200 = (s32 *) outer[-1].z;
    D_410200 = ext->unkA0;
    D_410200 = ext->unkA8;
    D_410200 = (s32 *) not[0xA].x;
    D_410200 = (s32 *) outer[0xA].x.x.x;
    D_410200 = (s32 *) outer[0xA].y.x;
    D_410200 = (s32 *) outer[0xA].z;
    func_004000B0(ext);
}

void Outer_Init(struct Extended *ext) {
    struct Outer *self = (struct Outer *) ext;
    self->x.x.x = 1;
    self->x.x.y = 2;
    self->x.y = 3;
    self->x.z = 4;
    self->y.x = 5;
    self->y.y = 6;
    self->z = 7;
}
