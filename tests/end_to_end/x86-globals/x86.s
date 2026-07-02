# Global loads/stores: absolute [symbol] accesses, a byte-sized global
# store, and scaled-index array accesses (read and write).
test:
    MOV EAX, [_counter]
    INC EAX
    MOV [_counter], EAX
    MOV byte ptr [_flag], 0x1
    MOV ECX, dword ptr [ESP + 0x4]
    MOV EDX, dword ptr [ECX*0x4 + _table]
    ADD EDX, EAX
    MOV dword ptr [ECX*0x4 + _table], EDX
    MOV EAX, dword ptr [ESP + 0x8]
    MOV EAX, dword ptr [EAX + ECX*0x8 + 0x10]
    RET

.data
_table:
    .word 0, 0, 0, 0, 0, 0, 0, 0
