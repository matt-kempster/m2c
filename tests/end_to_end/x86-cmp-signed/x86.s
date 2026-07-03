# Signed comparisons: jl/jg must round-trip to signed </> on s32 values.
# Real MSVC6 /Od output (three-way compare); MSVC re-evaluates cmp for each
# `if` rather than sharing flags, keeps an ebp frame, and materializes -1
# with the `or eax, -1` size peephole.
test:
    PUSH EBP
    MOV EBP, ESP
    MOV EAX, dword ptr [EBP + 0x8]
    CMP EAX, dword ptr [EBP + 0xc]
    JL .Lless
    MOV ECX, dword ptr [EBP + 0x8]
    CMP ECX, dword ptr [EBP + 0xc]
    JG .Lgreater
    XOR EAX, EAX
    JMP .Ldone
.Lgreater:
    MOV EAX, 0x1
    JMP .Ldone
.Lless:
    OR EAX, -0x1
.Ldone:
    POP EBP
    RET
