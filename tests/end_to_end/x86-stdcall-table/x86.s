# Undecorated stdcall Win32 import. Ghidra sometimes emits calls to system
# DLL functions through undecorated thunk names (`call _GlobalFree`) with no
# `@N` suffix and no import-slot decoration. The callee-cleanup byte count is
# recovered from a built-in table of well-known Win32/GDI/DirectX APIs
# (STDCALL_API_ARG_BYTES), so GlobalFree is known to pop its single argument.
# The loop back-edge forces the stack to balance across iterations, which only
# holds if GlobalFree is treated as stdcall (edi, a callee-saved register,
# carries the loop counter across the calls).
test:
    PUSH EDI
    PUSH ESI
    MOV ESI, dword ptr [ESP + 0xc]
    XOR EDI, EDI
_LAB_00401000:
    PUSH ESI
    CALL _GlobalFree
    ADD EDI, 0x1
    CMP EDI, 0xA
    JL _LAB_00401000
    POP ESI
    POP EDI
    RET
