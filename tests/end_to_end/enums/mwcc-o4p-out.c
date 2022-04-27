enum my_enum test(enum my_enum x) {
    if (x != TWO_TOO) {
        if (x < TWO_TOO) {
            if (x != ZERO) {
                return ZERO;
            }
            return array->unk0;
        }
        if (x != THREE) {
            return ZERO;
        }
        return array->unk8;
    }
    return array->unk4;
}
