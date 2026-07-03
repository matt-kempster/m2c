# A switch whose jump table is referenced by raw address rather than by a
# label of its own (`jmp [eax*4 + 0x401040]`). Ghidra still labels the jmp
# `_switchD_<id>_switchD` and the table's `.long` entries target this switch's
# `_switchD_<id>_caseD_*` labels, so X86JumpTablePattern recovers the table
# from that run of case entries and rewrites the jmp to reference it.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    CMP EAX, 0x3
    JA _switchD_00401000_caseD_default
_switchD_00401000_switchD:
    JMP dword ptr [EAX*0x4 + 0x401040]
_switchD_00401000_caseD_0:
    MOV EAX, 0x10
    RET
_switchD_00401000_caseD_1:
    MOV EAX, 0x20
    RET
_switchD_00401000_caseD_2:
    MOV EAX, 0x30
    RET
_switchD_00401000_caseD_3:
    MOV EAX, 0x40
    RET
_switchD_00401000_caseD_default:
    XOR EAX, EAX
    RET
.long _switchD_00401000_caseD_0
.long _switchD_00401000_caseD_1
.long _switchD_00401000_caseD_2
.long _switchD_00401000_caseD_3
