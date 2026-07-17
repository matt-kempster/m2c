# A pending memory argument must be captured before an intervening store
# changes the memory it reads.
test:
    PUSH dword ptr [_g]
    MOV dword ptr [_g], 0x5
    CALL _f
    ADD ESP, 0x4
    RET
