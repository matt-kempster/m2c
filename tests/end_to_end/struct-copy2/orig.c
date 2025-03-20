typedef struct Vec {
	float x;
	float y;
	float z;
} Vec;

typedef struct Test {
	int i;
	int j;
	unsigned char k;
} Test;


void test(Vec *a, Vec *b, Test *c, Test *d) {
	*a = *b;
	*c = *d;
}
