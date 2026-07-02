# Signed comparisons: jl/jg must round-trip to signed </> on s32 values.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    CMP EAX, dword ptr [ESP + 0x8]
    JL .Lless
    JG .Lgreater
    MOV EAX, 0x0
    RET
.Lless:
    MOV EAX, -0x1
    RET
.Lgreater:
    MOV EAX, 0x1
    RET
