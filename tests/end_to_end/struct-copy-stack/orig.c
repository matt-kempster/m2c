typedef struct Vec {
	float x;
	float y;
	float z;
} Vec;

void test2(Vec * b) {
	Vec a = *b;
	b->z = 4;
}

void test(Vec *b) {
	Vec a = {0};
	Vec c = a;
	c = *b;	
}