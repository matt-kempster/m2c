# Intel syntax spells a near return "retn", and a stdcall callee's "retn N" pops N
# argument bytes on the way out. Both must be treated like a plain "ret".
test:
    mov eax, dword ptr [esp + 0x4]
    add eax, dword ptr [esp + 0x8]
    retn 0x8
