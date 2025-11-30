#pragma push
#pragma pack(1)

typedef struct Test0 {
	int a;
	int b;
} Test0;

typedef struct Test1 {
	int a;
	int b;
	char c;
} Test1;

typedef struct Test2 {
	int a;
	int b;
	short c;
} Test2;

typedef struct Test3 {
	int a;
	int b;
	short c;
	char d;
} Test3;

typedef struct Test4 {
	int a;
	int b;
	int c;
} Test4;

typedef struct Test5 {
	int a;
	int b;
	int c;
	char d;
} Test5;

typedef struct Test6 {
	int a;
	int b;
	int c;
	short d;
} Test6;

typedef struct Test7 {
	int a;
	int b;
	int c;
	short d;
	char e;
} Test7;

Test7 s7;
Test7 d7;

#pragma pop

void test_0(Test0 *a, Test0 *b) {
	*a = *b;
}

void test_1(Test1 *a, Test1 *b) {
	*a = *b;
}

void test_2(Test2 *a, Test2 *b) {
	*a = *b;
}

void test_3(Test3 *a, Test3 *b) {
	*a = *b;
}

void test_4(Test4 *a, Test4 *b) {
	*a = *b;
}

void test_5(Test5 *a, Test5 *b) {
	*a = *b;
}

void test_6(Test6 *a, Test6 *b) {
	*a = *b;
}

void test_7(Test7 *a, Test7 *b) {
	*a = *b;
	d7 = s7;
}

void test() {}
