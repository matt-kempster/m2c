from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest

from m2c.arch_x86 import X86Arch, X86JumpTablePattern
from m2c.asm_file import AsmData, Label
from m2c.asm_instruction import AsmState, RegFormatter, parse_asm_instruction
from m2c.asm_pattern import AsmMatcher, BodyPart
from m2c.error import DecompFailure
from m2c.instruction import InstructionMeta


RAW = """\
_switchD_00401000_switchD:
    JMP dword ptr [EAX*0x4 + 0x401040]
_switchD_00401000_caseD_0:
    RET
_switchD_00401000_caseD_1:
    RET
.long _switchD_00401000_caseD_0
.long _switchD_00401000_caseD_1
"""


class TestGhidraFixJumpTables(unittest.TestCase):
    def test_script_labels_and_rewrites_raw_table(self) -> None:
        script = Path(__file__).parents[2] / "tools" / "ghidra_fix_jumptables.py"
        result = subprocess.run(
            [sys.executable, str(script)],
            input=RAW,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("JMP dword ptr [EAX*0x4 + _m2c_jtbl_401040]", result.stdout)
        self.assertIn("_m2c_jtbl_401040:\n.long", result.stdout)

    def test_raw_table_error_points_to_script(self) -> None:
        arch = X86Arch()
        state = AsmState(reg_formatter=RegFormatter())
        asm = parse_asm_instruction(
            "JMP dword ptr [EAX*0x4 + 0x401040]", arch, state
        )
        instr = arch.parse(asm.mnemonic, asm.args, InstructionMeta.missing())
        label = "_switchD_00401000_switchD"
        body: list[BodyPart] = [Label([label]), instr]
        matcher = AsmMatcher(arch, AsmData(), body, {label}, index=1)
        with self.assertRaises(DecompFailure) as caught:
            X86JumpTablePattern().match(matcher)
        self.assertIn("tools/ghidra_fix_jumptables.py", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
