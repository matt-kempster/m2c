# Arithmetic: add/sub/neg, 3-operand imul, cdq + idiv (/ and %),
# and unsigned mul through edx:eax.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    MOV ECX, dword ptr [ESP + 0x8]
    MOV EDX, dword ptr [ESP + 0xc]
    ADD EAX, ECX
    IMUL EDX, EDX, 0x3
    SUB EAX, EDX
    CDQ
    IDIV ECX
    NEG EDX
    ADD EAX, EDX
    MUL ECX
    ADD EAX, EDX
    RET
