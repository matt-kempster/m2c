# A direct mutable memory operand carried by the x87 status marker must be
# captured before an intervening store. The marker itself must disappear.
test:
    FLD dword ptr [ESP + 0x4]
    FCOMP dword ptr [g]
    MOV dword ptr [g], 0
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET
