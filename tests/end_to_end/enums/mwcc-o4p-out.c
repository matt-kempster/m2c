enum my_enum test(enum my_enum x) {
    if (x != 2) {
        if (x < 2) {
            if (x != 0) {
                return ZERO;
            }
            return array->unk0;
        }
        if (x != 3) {
            return ZERO;
        }
        return array->unk8;
    }
    return array->unk4;
}
