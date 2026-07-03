# Arithmetic: add/sub/neg, immediate imul (Ghidra prints the imul r,r/m,imm
# encoding with three operands), cdq + idiv (/ and %), and unsigned mul
# through edx:eax. Real MSVC6 /O1 output: each of / and % gets its own
# cdq + idiv pair (MSVC never reuses one idiv for both), *0x8c stays an
# immediate imul at /O1 (at /O2 it becomes an lea/shl chain), and esi/edi
# are saved with pushes without an ebp frame.
test:
    MOV EAX, dword ptr [ESP + 0xc]
    MOV ECX, dword ptr [ESP + 0x4]
    IMUL EAX, EAX, 0x8c
    PUSH ESI
    MOV ESI, dword ptr [ESP + 0xc]
    SUB ECX, EAX
    PUSH EDI
    ADD ECX, ESI
    MOV EAX, ECX
    CDQ
    IDIV ESI
    MOV EAX, EDX
    NEG EAX
    MUL ESI
    MOV EAX, ECX
    MOV EDI, EDX
    CDQ
    IDIV ESI
    ADD EAX, EDI
    POP EDI
    POP ESI
    RET
