# The double-returning counterpart of x86-fpu-wrapper-float:
# `double test(void) { return returns_double(); }` as a bare `call; ret`.
# The context declaration `double returns_double(void);` seeds a +1 x87 stack
# delta so the FPU prepass runs (despite no local FPU mnemonic) and models the
# callee's st(0) return, which `test` forwards.
test:
    CALL _returns_double
    RET
