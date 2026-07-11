test:
    MOV EAX, dword ptr [_data]
    PUSH EAX
    CALL "_stdcall@4"
    CALL _callee
    CALL __crt
    CALL _foo
    CALL foo
    RET

.section .data
_data:
    .long 1
