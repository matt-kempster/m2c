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
    IntPair ints;
    FloatPair floats;
};

struct Container {
    union SameSizeUnion data;
};

extern int test_func(struct Container *arg);
