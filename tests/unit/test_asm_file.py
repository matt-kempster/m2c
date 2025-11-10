from __future__ import annotations

import io
import unittest

from m2c.arch_mips import MipsArch
from m2c.asm_file import parse_file
from m2c.main import parse_flags


class TestAsmDirectives(unittest.TestCase):
    def test_set_string_ignored(self) -> None:
        asm_source = """.text
.set ___C__0L_HOAH_fc3_m1_4lls__AA_, "??_C@_0L@HOAH@fc3_m1?4lls?$AA@"
test_label:
    nop
"""
        arch = MipsArch()
        options = parse_flags(
            [
                "-t",
                "mips-ido-c",
                "test.s",
            ]
        )
        buffer = io.StringIO(asm_source)
        buffer.name = "test.s"  # type: ignore[attr-defined]
        parse_file(buffer, arch, options)


if __name__ == "__main__":
    unittest.main()
