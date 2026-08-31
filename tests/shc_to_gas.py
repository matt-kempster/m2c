#!/usr/bin/env python3
"""Convert SHC assembly to GNU as format.
SHC's directives, comment markers, hex literals, and a couple of mnemonics are
problematic

Example:

        .EXPORT _test
        .SECTION    P,CODE,ALIGN=4
    _test:                  ; function: test
        MOV.L       L238,R1
        FMOV.S      FR4,FR5
        RTS
        NOP
        .SECTION    D,DATA,ALIGN=4
    L238:
        .DATA.L     H'1234
        .END

becomes:

    .globl _test
    .section .text,"ax",@progbits
    .balign 4
    _test:                   ! function: test
        MOV.L       L238,R1
        FMOV FR4,FR5
        RTS
        NOP
    .section .data,"aw",@progbits
    .balign 4
    L238:
    .long 0x1234
"""

import re
import sys
from typing import List, Optional, Tuple


def hex_converted(text: str) -> str:
    """convert H'1234 into GNU as syntax 0x1234"""
    return re.sub(r"h'(?=[0-9a-f])", "0x", text, flags=re.IGNORECASE)


def convert_fmov_reg_reg(raw: str) -> str:
    """FMOV.S FRn,FRm -> FMOV FRn,FRm"""
    return re.sub(
        r"\bFMOV\.S(\s+FR\d+\s*,\s*FR\d+)",
        r"FMOV\1",
        raw,
        flags=re.IGNORECASE,
    )


SECTION_NAMES: List[Tuple[str, str]] = [
    ("P", ".text"),
    ("C", ".rodata"),
    ("D", ".data"),
    ("B", ".bss"),
]


def gas_section_name(name: str) -> str:
    for shc_name, gas_name in SECTION_NAMES:
        if name.upper() == shc_name:
            return gas_name

    return name


def parse_uint(s: str) -> Optional[int]:
    s = s.strip()

    if not s:
        return None

    if not all(c.isdigit() for c in s):
        return None

    return int(s, 10)


def convert_line(line: str) -> str:
    line = line.rstrip("\r")

    line = line.replace(";", " !", 1)

    s = line.strip()

    if not s:
        return ""

    parts = s.split(None, 1)
    directive = parts[0].upper()
    rest = parts[1] if len(parts) > 1 else ""

    if directive in (".EXPORT", ".IMPORT") and rest:
        is_export = directive == ".EXPORT"
        return (".globl " if is_export else ".extern ") + rest

    if directive == ".SECTION" and rest:
        parts = [p.strip() for p in rest.split(",")]
        assert 1 <= len(parts) <= 3, s
        if len(parts) == 1:
            if not any(c in " \t\r\n," for c in rest):
                flags = ',"ax",@progbits' if rest.upper() == "P" else ',"aw",@progbits'

                return ".section " + gas_section_name(rest) + flags

        else:
            name = parts[0]
            kind = parts[1]

            if name and kind.upper() in ("CODE", "DATA"):
                result = ".section " + gas_section_name(name)

                if kind.upper() == "CODE":
                    result += ',"ax",@progbits'
                else:
                    result += ',"aw",@progbits'

                if len(parts) == 3:
                    tail = parts[2]

                    if tail.upper().startswith("ALIGN="):
                        align = parse_uint(tail[6:])

                        if align is not None:
                            result += "\n.balign " + str(align)

                return result

    if directive in (".DATA.B", ".DATA.W", ".DATA.L"):
        directive = {"B": ".byte", "W": ".word", "L": ".long"}[directive[-1]]
        return directive + " " + hex_converted(rest)

    if directive in (".RES.B", ".RES.W", ".RES.L"):
        wsize = {"B": 1, "W": 2, "L": 4}[directive[-1]]
        return f".space {wsize} * {hex_converted(rest)}"

    if directive == ".END":
        return ""

    if directive.startswith("."):
        raise ValueError("unsupported SHC directive: " + line)

    return hex_converted(convert_fmov_reg_reg(line))


def convert(text: str) -> str:
    text = text.replace("\n+", "")
    return "".join(convert_line(line) + "\n" for line in text.splitlines())


def main() -> int:
    try:
        sys.stdout.write(convert(sys.stdin.read()))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
