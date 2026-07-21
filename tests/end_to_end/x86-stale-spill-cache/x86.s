# A registerless overwrite must invalidate the prior register-backed spill
# cache entry for the same stack slot.
test:
    PUSH EBP
    MOV EBP, ESP
    SUB ESP, 0x8
    MOV EAX, dword ptr [EBP + 0x8]
    ADD EAX, 1
    MOV dword ptr [EBP - 0x8], EAX
    CALL _foo
    MOV dword ptr [EBP - 0x8], 1234
    CALL _bar
    MOV EAX, dword ptr [EBP - 0x8]
    MOV ESP, EBP
    POP EBP
    RET
