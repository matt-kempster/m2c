# A nested float intrinsic in an x87 compare operand must be materialized
# before the intervening store, while the status-word marker itself disappears.
test:
    FLD dword ptr [ESP + 0x4]
    FSQRT
    FCOMP dword ptr [ESP + 0x8]
    MOV dword ptr [ESP + 0xc], 1
    FNSTSW AX
    TEST AH, 0x41
    SETZ AL
    MOVZX EAX, AL
    RET
