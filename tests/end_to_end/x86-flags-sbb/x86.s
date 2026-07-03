# sbb sets subtract-with-borrow flags, not logic-op flags. This is the MSVC
# lowering of an unsigned 64-bit less-than: `sub` the low halves (producing a
# borrow) then `sbb` the high halves; the resulting carry flag is the borrow
# out of the full 64-bit subtraction, i.e. the unsigned comparison. `setb`
# reads that carry. With sbb modeled as logic (the bug), the carry would be a
# constant 0 and the comparison would collapse.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    MOV EDX, dword ptr [ESP + 0x8]
    SUB EAX, dword ptr [ESP + 0xc]
    SBB EDX, dword ptr [ESP + 0x10]
    SETB AL
    MOVZX EAX, AL
    RET
