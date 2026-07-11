# A Ghidra raw-address switch after tools/ghidra_fix_jumptables.py cleanup.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    CMP EAX, 0x3
    JA _switchD_00401000_caseD_default
_switchD_00401000_switchD:
    JMP dword ptr [EAX*0x4 + _m2c_jtbl_401040]
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
_m2c_jtbl_401040:
.long _switchD_00401000_caseD_0
.long _switchD_00401000_caseD_1
.long _switchD_00401000_caseD_2
.long _switchD_00401000_caseD_3
