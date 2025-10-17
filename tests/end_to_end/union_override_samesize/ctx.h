// Test with union fields of the same size
typedef struct {
    int x;
    int y;
} IntPair;

typedef struct {
    float a;
    float b;
} FloatPair;

union SameSizeUnion {
    FloatPair floats;
    IntPair ints;
};

struct Container {
    union SameSizeUnion data;
};

extern int test(struct Container *arg);
