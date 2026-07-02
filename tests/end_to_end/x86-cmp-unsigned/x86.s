# Unsigned comparisons: ja/jb/jbe must decompile to u32 compares.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    MOV ECX, dword ptr [ESP + 0x8]
    CMP EAX, ECX
    JA .Labove
    JB .Lbelow
    MOV EAX, 0x0
    RET
.Labove:
    CMP EAX, 0x100
    JBE .Lbelow
    MOV EAX, 0x1
    RET
.Lbelow:
    MOV EAX, 0x2
    RET
