# Nested calls: the result of the inner call is used as an argument to the
# outer call. _outer(_inner(arg0), arg1).
test:
    MOV EAX, dword ptr [ESP + 0x4]
    PUSH EAX
    CALL _inner
    ADD ESP, 0x4
    MOV ECX, dword ptr [ESP + 0x8]
    PUSH ECX
    PUSH EAX
    CALL _outer
    ADD ESP, 0x8
    RET
