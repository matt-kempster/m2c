# MSVC structured-exception-handling (SEH) prologue/epilogue. After the
# standard `push ebp; mov ebp, esp` frame setup, the function installs a
# 16-byte exception registration record at fs:[0] and restores the previous
# chain head at fs:[0] on the way out. None of this is visible at the C
# level, so the SEH bookkeeping is recognized and dropped (see X86SehPattern):
# the install becomes a bare frame allocation and the fs:[0] restore is
# elided, leaving just the function body.
test:
    PUSH EBP
    MOV EBP, ESP
    PUSH -0x1
    PUSH offset _scopetable
    PUSH offset _except_handler3
    MOV EAX, FS:[0x0]
    PUSH EAX
    MOV dword ptr FS:[0x0], ESP
    SUB ESP, 0x8
    MOV EAX, dword ptr [EBP + 0x8]
    ADD EAX, dword ptr [EBP + 0xc]
    MOV ECX, dword ptr [EBP + -0x10]
    MOV dword ptr FS:[0x0], ECX
    MOV ESP, EBP
    POP EBP
    RET
