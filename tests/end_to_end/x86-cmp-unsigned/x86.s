# Unsigned comparisons: ja/jb/jbe must decompile to u32 compares.
# Real MSVC6 /Od output; each `if` gets its own cmp, arguments are read
# through an ebp frame, and the immediate compare uses `cmp mem, imm`.
test:
    PUSH EBP
    MOV EBP, ESP
    MOV EAX, dword ptr [EBP + 0x8]
    CMP EAX, dword ptr [EBP + 0xc]
    JA .Labove
    MOV ECX, dword ptr [EBP + 0x8]
    CMP ECX, dword ptr [EBP + 0xc]
    JB .Lbelow
    XOR EAX, EAX
    JMP .Ldone
.Lbelow:
    MOV EAX, 0x2
    JMP .Ldone
.Labove:
    CMP dword ptr [EBP + 0x8], 0x100
    JBE .Lsmall
    MOV EAX, 0x1
    JMP .Ldone
.Lsmall:
    MOV EAX, 0x2
.Ldone:
    POP EBP
    RET
