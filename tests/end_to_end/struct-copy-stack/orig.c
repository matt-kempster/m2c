#pragma push
#pragma pack(1)

typedef struct Vec {
	float x;
	float y;
	float z;
} Vec;

void test() {
	Vec a = {0};
	Vec c = a;
	Vec b;
	b.x = 4;
	c = b;
}