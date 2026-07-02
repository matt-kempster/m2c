# movsx/movzx byte and word loads, through globals (so the s8/u8/s16/u16
# types are visible in the declarations) and through a pointer argument.
test:
    MOVSX EAX, byte ptr [_sbyte]
    MOVZX ECX, byte ptr [_ubyte]
    ADD EAX, ECX
    MOVSX ECX, word ptr [_sword]
    ADD EAX, ECX
    MOVZX ECX, word ptr [_uword]
    ADD EAX, ECX
    MOV EDX, dword ptr [ESP + 0x4]
    MOVSX ECX, byte ptr [EDX + 0x1]
    ADD EAX, ECX
    RET
