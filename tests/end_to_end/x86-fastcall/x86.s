# __fastcall register arguments: MSVC passes the first two register-sized
# arguments in ecx and edx. Reading them before any write registers them as
# arguments (compiler-generated cdecl/stdcall code never reads these
# caller-save registers uninitialized), instead of failing with
# unset-register errors. Extra arguments still come from the stack, after
# the register ones.
test:
    MOV EAX, ECX
    ADD EAX, EDX
    ADD EAX, dword ptr [ESP + 0x4]
    RET 0x4
