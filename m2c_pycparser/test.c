typedef int typeid;

const __attribute__((a1)) __attribute__((a2)) const int const __attribute__((b1, b2, b3(arg), b4, ,)) __attribute__((b5)) const __attribute__(()) x;

int __attribute__((a))
	foo asm("a" "b") __attribute__((b1)),
	bar asm("c") __attribute__((b2)),
	baz __attribute__((b3)) = 2;

enum A {
	x __attribute__((a)) = 1,
	y __attribute__((a))
};

__attribute__((a1))
struct __attribute__((b1)) A { } __attribute__((b2))
	x __attribute__((a2));

// int (__attribute__((a)) *x); // unsupported
__attribute__((a)) int *y;
// int *__attribute__((a)) z; // unsupported
int *w __attribute__((a));

int f(int x) __attribute__((a));

void f(__attribute__((unused)) int x __attribute__((unused)));

// void f(int ar[static __attribute__((a)) 10]); // unsupported

int arr[] = { [1 ... 2] = 0, };

__attribute__((a)) int f(x) int x; {
	switch (1) {
	case 1 ... 2:
		__attribute__((fallthrough));
	typeid:;
	}
	typeof(1) x = 1;
	typeof(int) x = 1;
	(int)x = 1;

	asm __volatile__ __inline__("");
	asm("");
	asm("" :);
	asm("" ::);
	asm("" :::);
	asm("" ::::);
	asm goto("" : [a]"a"(1) : "a"(1), [typeid]"b"(2): "a", "b" : a, typeid);
}
