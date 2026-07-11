# A tail call: `jmp` to a label outside the function becomes a
# return-with-call. The `_LAB_...` label name keeps the fall-through block in
# the function (Ghidra x86 export convention).
test:
    MOV EAX, dword ptr [ESP + 0x4]
    TEST EAX, EAX
    JZ _LAB_00401010
    JMP _other_function
_LAB_00401010:
    XOR EAX, EAX
    RET
