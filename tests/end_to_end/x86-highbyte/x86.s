# High-byte reads are extracted from bits 8..15 of their full registers.
test:
    MOVZX EAX, word ptr [ESP + 0x4]
    MOVZX ECX, AH
    MOV EBX, dword ptr [ESP + 0x8]
    ADD CL, BH
    MOVZX EAX, CL
    RET
