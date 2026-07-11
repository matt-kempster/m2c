# A scaled-index store has no base register, so its destination is a generic
# address expression rather than an AddressMode (see mem_store).
test:
    MOV EAX, dword ptr [ESP + 0x4]
    MOV ECX, dword ptr [ESP + 0x8]
    MOV dword ptr [ECX*4 + _table], EAX
    MOV byte ptr [ECX + EAX*2 + 0x1], 0x7
    RET
