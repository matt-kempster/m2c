# A tail call: `jmp` to a label outside the function becomes a
# return-with-call. Ghidra's x86 output likewise labels the local fall-through
# block.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    TEST EAX, EAX
    JZ .Lreturn_zero
    JMP _other_function
.Lreturn_zero:
    XOR EAX, EAX
    RET
