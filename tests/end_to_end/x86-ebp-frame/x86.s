# A function with a classic ebp frame: saved ebp, a saved register (esi),
# arguments at [ebp+8]/[ebp+0xc], and locals at [ebp-4]/[ebp-8].
test:
    PUSH EBP
    MOV EBP, ESP
    SUB ESP, 0x8
    PUSH ESI
    MOV ESI, dword ptr [EBP + 0x8]
    MOV dword ptr [EBP - 0x4], ESI
    MOV EAX, dword ptr [EBP + 0xc]
    ADD ESI, EAX
    MOV dword ptr [EBP - 0x8], ESI
    MOV EAX, dword ptr [EBP - 0x4]
    IMUL EAX, dword ptr [EBP - 0x8]
    POP ESI
    MOV ESP, EBP
    POP EBP
    RET
