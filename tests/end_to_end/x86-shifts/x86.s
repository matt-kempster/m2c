# Shifts: shl by an immediate (multiplication), sar (signed >>),
# shr (unsigned >>), and a variable shift through cl.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    MOV EDX, dword ptr [ESP + 0x8]
    SHL EAX, 0x2
    SAR EAX, 0x1
    SHR EDX, 0x3
    ADD EAX, EDX
    MOV ECX, dword ptr [ESP + 0xc]
    MOV EDX, dword ptr [ESP + 0x4]
    SHL EDX, CL
    ADD EAX, EDX
    RET
