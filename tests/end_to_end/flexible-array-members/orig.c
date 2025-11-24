struct A {
	int x;
	int ys[];
};

struct B {
	int x, y;
};

struct C {
	int x;
	struct B bs[];
};

int test(struct A *a, struct C *c, int i) {
	return a->ys[i] + c->bs[i].y;
}
