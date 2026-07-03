# A stdcall callee: it pops its own two arguments on return (`ret 8`).
test:
    MOV EAX, dword ptr [ESP + 0x4]
    MOV ECX, dword ptr [ESP + 0x8]
    ADD EAX, ECX
    RET 0x8
