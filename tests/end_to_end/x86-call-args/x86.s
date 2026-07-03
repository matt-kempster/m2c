# cdecl call with pushed arguments and an `add esp, N` cleanup.
# Arguments are pushed right-to-left, so the last push is arg0:
# _add_two(arg0, arg1).
test:
    MOV EAX, dword ptr [ESP + 0x4]
    MOV ECX, dword ptr [ESP + 0x8]
    PUSH ECX
    PUSH EAX
    CALL _add_two
    ADD ESP, 0x8
    RET
