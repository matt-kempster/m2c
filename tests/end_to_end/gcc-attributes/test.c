struct __attribute__((aligned(0x10))) A {
	int x;
};

struct ContainsA {
	int x;
	struct A a;
};

struct B {
	int x;
} __attribute__((aligned(0x10)));

#1 "Lower-than default alignment attribute has no effect without pack"
struct C {
	int x;
} __attribute__((aligned(1)));

struct D {
	__attribute__((aligned(1))) int x;
};

typedef __attribute__((aligned(1))) int underaligned;

struct E {
	underaligned y;
};

struct F {
	int x;
	int y __attribute__((aligned(0x10))) __attribute__((aligned(0x20)));
};

struct G {
	int x;
	_Alignas(0x10) int y;
};

__attribute__((aligned(1))) int underaligned_var;
underaligned underaligned_var2;
