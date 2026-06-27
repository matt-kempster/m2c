struct U8 {
	u8 x;
};

struct U64 {
	u64 x;
};

struct __attribute__((packed, aligned(2))) Packed {
	char x;
	int y;
};

struct __attribute__((packed)) Packed2 {
	char x;
	int y;
};

struct PackedMember {
	char x;
	__attribute__((packed))
	int y;
};
