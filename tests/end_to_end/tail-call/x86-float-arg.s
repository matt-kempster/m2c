test:
    FLD dword ptr [ESP + 0x4]
    PUSH ECX
    FSTP dword ptr [ESP]
    JMP _consume_float
