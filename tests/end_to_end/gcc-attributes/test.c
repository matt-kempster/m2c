struct __attribute__((aligned(0x10))) Overaligned1 {
	int x;
};

struct ContainsOveraligned {
	int x;
	struct Overaligned1 a;
	int y;
};

struct Overaligned2 {
	int x;
} __attribute__((aligned(0x10)));

struct UnderalignedNoop {
	int x;
} __attribute__((aligned(1)));

struct UnderalignedMemberNoop {
	__attribute__((aligned(1))) int x;
};

typedef __attribute__((aligned(1))) int underaligned;
typedef __attribute__((aligned(0x10))) int overaligned;

struct UnderalignedMemberViaTypedef {
	underaligned y;
};

struct OveralignedMemberViaTypedef {
	char a;
	overaligned y;
	char c;
};

struct MultipleAttrs {
	int x;
	int y __attribute__((aligned(0x10))) __attribute__((aligned(0x20)));
};

struct Alignas {
	int x;
	_Alignas(0x10) int y;
};

__attribute__((aligned(1))) int underaligned_var;
underaligned underaligned_var2;

__attribute__((aligned(0x10))) int overaligned_var;
overaligned overaligned_var2;

struct __attribute__((packed, aligned(2))) Packed {
	char x;
	int y;
};

struct PackedMember {
	char x;
	__attribute__((packed))
	int y;
};

#pragma pack(push,1)
struct PragmaPack1 {
	char a;
	__attribute__((aligned(0x10))) int nicetry;
};
#pragma pack(2)
struct PragmaPack2 {
	char a;
	__attribute__((aligned(0x10))) int nicetry;
};
#pragma pack(0)
struct PragmaPackNone1 {
	char a;
	int b;
};
#pragma pack()
struct PragmaPackNone2 {
	char a;
	int b;
};
#pragma pack(pop)

struct __attribute__((packed, aligned(2))) PackedWithOveralignedMember {
	char x;
	int y;
	__attribute__((aligned(0x10))) int z;
};

typedef int long128 __attribute__ ((mode (TI)));
typedef unsigned int u_long128 __attribute__ ((mode (TI)));

long128 i128;
u_long128 u128;

struct I128s {
	long128 a;
	u_long128 b;
};
