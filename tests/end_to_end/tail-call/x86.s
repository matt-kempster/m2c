# A tail call: `jmp` to a label outside the function becomes a
# return-with-call.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    TEST EAX, EAX
    JZ .Lreturn_zero
    JMP _other_function
.Lreturn_zero:
    XOR EAX, EAX
    RET
