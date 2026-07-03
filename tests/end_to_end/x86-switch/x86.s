# A switch implemented via an indirect jump through a jump table. Label names
# follow the Ghidra x86 export convention (_switchD_.../_caseD_...) so the
# parser keeps the case blocks within the function.
test:
    MOV EAX, dword ptr [ESP + 0x4]
    CMP EAX, 0x3
    JA _switchD_00401000_caseD_default
    JMP dword ptr [EAX*0x4 + _switchD_00401000_switchdataD_00401040]
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
_switchD_00401000_switchdataD_00401040:
    .long _switchD_00401000_caseD_0, _switchD_00401000_caseD_1, _switchD_00401000_caseD_2, _switchD_00401000_caseD_3
