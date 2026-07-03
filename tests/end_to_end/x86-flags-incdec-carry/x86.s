# inc/dec preserve the carry flag. Hand-optimized code can set the carry with
# an add and then branch on it after a dec, whose zero flag combines with the
# preserved carry: `jbe` after `dec` means "carry (from the add) OR the dec
# result is zero". Modeling dec's composite unsigned predicate as if carry
# were clear (the bug) would drop the carry term.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    ADD EAX, dword ptr [ESP + 0x8]
    DEC EAX
    JBE .Lbe
    MOV EAX, 0x0
    RET
.Lbe:
    MOV EAX, 0x1
    RET
