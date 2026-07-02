# A for-loop with a backwards conditional branch: cmp/jl at the bottom,
# a scaled-index store, and inc as the induction step.
test:
    MOV ECX, dword ptr [ESP + 0x4]
    MOV EDX, dword ptr [ESP + 0x8]
    TEST ECX, ECX
    JLE .Ldone
    XOR EAX, EAX
.Lloop:
    MOV dword ptr [EDX + EAX*0x4], EAX
    INC EAX
    CMP EAX, ECX
    JL .Lloop
.Ldone:
    RET
