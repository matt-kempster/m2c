# Undecorated stdcall Win32 import. Disassemblers sometimes emit calls to system
# DLL functions through undecorated thunk names (`call _GlobalFree`) with no
# `@N` suffix and no import-slot decoration. The callee-cleanup byte count is
# recovered structurally: the first stack-rewrite pass assumes cdecl, the loop
# back-edge then sees conflicting stack depths (the pushed argument is never
# cleaned by the caller), and the infer_direct_stdcall retry reclassifies the
# call as popping its own 4 argument bytes. The loop forces the stack to
# balance across iterations, which only holds if GlobalFree is treated as
# stdcall (edi, a callee-saved register, carries the loop counter across the
# calls).
test:
    PUSH EDI
    PUSH ESI
    MOV ESI, dword ptr [ESP + 0xc]
    XOR EDI, EDI
.Lloop:
    PUSH ESI
    CALL _GlobalFree
    ADD EDI, 0x1
    CMP EDI, 0xA
    JL .Lloop
    POP ESI
    POP EDI
    RET
