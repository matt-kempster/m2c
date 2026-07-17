# A pending memory argument for an outer call survives the intervening inner
# call and must keep the value read before that call.
test:
    PUSH dword ptr [_g]
    MOV EAX, dword ptr [ESP + 0x8]
    PUSH EAX
    CALL _inner
    ADD ESP, 0x4
    PUSH EAX
    CALL _outer
    ADD ESP, 0x8
    RET
