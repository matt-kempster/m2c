#!/usr/bin/env python3
"""Label raw-address Ghidra x86 jump tables for m2c."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import List, Optional, Tuple


SWITCH_LABEL = re.compile(r"^\s*(_switchD_(\w+?)_switchD):\s*$")
CASE_LABEL = re.compile(r"^\s*_switchD_(\w+?)_(?:caseD_\w+|default):\s*$")
JUMP = re.compile(r"(?i)\bjmp\b[^\[]*\[[^\]]*\]")
HEX_LITERAL = re.compile(r"(?i)\b0x[0-9a-f]+\b")
LONG = re.compile(r"^\s*\.long\s+(.+?)\s*(?:[#;].*)?$")


def switch_id_near(lines: List[str], index: int) -> Optional[str]:
    if index > 0:
        match = SWITCH_LABEL.match(lines[index - 1])
        if match:
            return match.group(2)
    for line in lines[index + 1 : index + 3]:
        match = CASE_LABEL.match(line)
        if match:
            return match.group(1)
    return None


def long_symbols(line: str) -> Optional[List[str]]:
    match = LONG.match(line)
    if not match:
        return None
    return [item.strip() for item in match.group(1).split(",")]


def table_run(lines: List[str], switch_id: str) -> Optional[Tuple[int, int]]:
    own = re.compile(rf"^_switchD_{re.escape(switch_id)}_(?:caseD_\w+|default)$")
    shared_default = re.compile(r"^_switchD_\w+_default$")
    runs: List[Tuple[int, int]] = []
    start: Optional[int] = None
    has_case = False
    entries = 0
    for index, line in enumerate(lines + [""]):
        symbols = long_symbols(line)
        valid = symbols is not None and all(
            own.match(symbol) or shared_default.match(symbol) for symbol in symbols
        )
        if valid:
            if start is None:
                start = index
                has_case = False
                entries = 0
            has_case |= any("_caseD_" in symbol for symbol in symbols or [])
            entries += len(symbols or [])
        elif start is not None:
            # A single-entry run is not a jump table; refuse it like the
            # in-decompiler pattern does for labeled tables.
            if has_case and entries >= 2:
                runs.append((start, index))
            start = None
    return runs[0] if len(runs) == 1 else None


def fix_text(text: str) -> str:
    lines = text.splitlines(keepends=True)
    insertions: List[Tuple[int, str]] = []
    for index, line in enumerate(lines):
        jump = JUMP.search(line)
        if not jump:
            continue
        addresses = [
            match
            for match in HEX_LITERAL.finditer(jump.group(0))
            if int(match.group(0), 16) > 0xFFFF
        ]
        if len(addresses) != 1:
            continue
        switch_id = switch_id_near(lines, index)
        if switch_id is None:
            continue
        run = table_run(lines, switch_id)
        if run is None:
            raise ValueError(f"cannot find unique jump table for switch {switch_id}")
        address_match = addresses[0]
        address = address_match.group(0)
        label = f"_m2c_jtbl_{int(address, 16):x}"
        start = jump.start() + address_match.start()
        end = jump.start() + address_match.end()
        lines[index] = line[:start] + label + line[end:]
        if not any(existing == label for _, existing in insertions):
            insertions.append((run[0], label))
    for index, label in sorted(insertions, reverse=True):
        lines.insert(index, f"{label}:\n")
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--in-place", action="store_true")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    if args.in_place:
        if not args.files:
            parser.error("--in-place requires at least one file")
        for path in args.files:
            path.write_text(fix_text(path.read_text()))
        return 0
    if len(args.files) > 1:
        parser.error("at most one input file is supported without --in-place")
    text = args.files[0].read_text() if args.files else sys.stdin.read()
    sys.stdout.write(fix_text(text))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
