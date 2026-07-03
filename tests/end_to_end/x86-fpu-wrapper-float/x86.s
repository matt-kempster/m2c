# A forwarding wrapper `float test(void) { return returns_float(); }`.
# The callee returns its float in st(0), but `test` has no x87 instruction of
# its own -- the FPU prepass would normally skip a function with no FPU
# mnemonic. The context declaration `float returns_float(void);` seeds a +1
# x87 stack delta for the callee, which both runs the pass and annotates the
# call as producing st(0), so the value is returned from `test` as a float.
test:
    CALL _returns_float
    RET
