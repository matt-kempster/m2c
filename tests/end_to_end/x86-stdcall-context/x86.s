# Undecorated stdcall call. MSVC calls a __stdcall function through a thunk
# symbol with no `@N` name decoration (`call _MyApiCall`), and `_MyApiCall`
# is not one of the built-in Win32 APIs, so the callee-pops-its-arguments
# convention is supplied by a context prototype marked
# __attribute__((stdcall)) (see orig.c). The callee pops its three arguments
# itself, so there is no caller-side `add esp` after the call.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    PUSH 0x30
    PUSH offset _caption
    PUSH EAX
    CALL _MyApiCall
    RET
