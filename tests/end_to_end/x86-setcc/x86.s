# setcc: materializing comparison results as 0/1 bytes (with the usual
# xor-zeroing of the full register beforehand).
test:
    MOV EAX, dword ptr [ESP + 0x4]
    MOV EDX, dword ptr [ESP + 0x8]
    XOR ECX, ECX
    CMP EAX, EDX
    SETL CL
    XOR EAX, EAX
    TEST EDX, EDX
    SETNZ AL
    ADD EAX, ECX
    RET
