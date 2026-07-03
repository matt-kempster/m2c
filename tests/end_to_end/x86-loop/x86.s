# A for-loop with a backwards conditional branch: cmp/jl at the bottom,
# a scaled-index store, and inc as the induction step. Real MSVC6 /O2
# output for `for (i = 0; i < n; i++) arr_ptr[i] = i;` with a global
# pointer: possible aliasing forces the base pointer to be reloaded every
# iteration, which keeps the store scaled-index instead of strength-reduced
# to a walking pointer.
test:
    MOV ECX, dword ptr [ESP + 0x4]
    XOR EAX, EAX
    TEST ECX, ECX
    JLE .Ldone
.Lloop:
    MOV EDX, dword ptr [_arr_ptr]
    MOV dword ptr [EDX + EAX*0x4], EAX
    INC EAX
    CMP EAX, ECX
    JL .Lloop
.Ldone:
    RET
