# `repne scasb` with a non-zero search byte: the memchr idiom. The compiler
# scans [edi..] for the byte in al (here 0x3b, ';') with ecx as a down-counter
# starting at -1, leaving edi one past the match; `not ecx; dec ecx` then turns
# the decremented counter into the number of bytes scanned before the match.
# Modeled as M2C_MEMCHR (a zero search byte would instead be the strlen idiom).
test:
    MOV EDI, dword ptr [ESP + 0x4]
    MOV AL, 0x3B
    MOV ECX, 0xFFFFFFFF
    REPNE SCASB
    NOT ECX
    DEC ECX
    MOV EAX, ECX
    RET
