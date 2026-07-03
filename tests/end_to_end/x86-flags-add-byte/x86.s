# A byte add followed by an unsigned above test (ja). The carry-out and the
# composite unsigned-above predicate must be computed at the 8-bit operand
# width, not 32-bit: `ja` here means "the 8-bit sum did not carry out and is
# nonzero". A width-unaware helper would model the carry/overflow as if the
# add were 32-bit.
test:
    MOV AL, byte ptr [ESP + 0x4]
    ADD AL, byte ptr [ESP + 0x8]
    JA .Labove
    MOV EAX, 0x0
    RET
.Labove:
    MOV EAX, 0x1
    RET
