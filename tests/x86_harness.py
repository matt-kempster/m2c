#!/usr/bin/env python3
"""Corpus harness for the x86 backend: run full decompilation (parsing,
flow-graph construction, and translation) over a directory of Ghidra-exported
.asm files, and report how many functions decompile without DecompFailure,
with a histogram of the remaining failure messages."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m2c.arch_x86 import X86Arch
from m2c.asm_file import AsmData, parse_file
from m2c.c_types import build_typemap
from m2c.instruction import Instruction
from m2c.main import build_global_info, decompile_function, parse_flags
from m2c.types import TypePool

DEFAULT_ASM_ROOT = Path.home() / "Projects/legoland/port2/project/asm"

RE_HEX = re.compile(r"0x[0-9a-fA-F]+|\b-?\d+\b")
RE_LABEL = re.compile(r"_(?:LAB|LLS|FUN|DAT|PTR|switchD)_?\w*")
RE_UNIMPLEMENTED = re.compile(
    r"x86 instruction evaluation is not implemented yet \(([^)]*)\): (\S+)"
)


def normalize(message: str) -> str:
    """Strip numbers and label/symbol names so messages bucket together."""
    return RE_HEX.sub("N", RE_LABEL.sub("L", message))


def classify(exc: Exception) -> str:
    """Normalize a failure into a histogram bucket."""
    from m2c.instruction import InstrProcessingFailure

    if isinstance(exc, InstrProcessingFailure) and isinstance(exc.__cause__, Exception):
        mnemonic = exc.instr.mnemonic
        exc = exc.__cause__
        m = RE_UNIMPLEMENTED.search(str(exc))
        if m:
            return f"unimplemented ({m.group(1)}): {mnemonic}"
        return f"{mnemonic}: " + normalize(str(exc).splitlines()[0])
    message = str(exc) or type(exc).__name__
    # Use the first interesting line, with numbers stripped.
    for line in message.splitlines():
        line = line.strip()
        if line:
            return normalize(line)
    return "<empty>"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asm-root", type=Path, default=DEFAULT_ASM_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top", type=int, default=25)
    cli_args = parser.parse_args()

    asm_files = sorted(cli_args.asm_root.glob("*.asm"))
    if cli_args.limit is not None:
        asm_files = asm_files[: cli_args.limit]
    if not asm_files:
        print(f"No .asm files found under {cli_args.asm_root}")
        return

    arch = X86Arch()
    # --stop-on-error makes translation failures raise instead of degrading
    # into `/* Error */` comments, so they can be tallied.
    options = parse_flags(["-t", "x86", "--stop-on-error", "dummy.asm"])
    typemap = build_typemap([], arch, use_cache=False)

    files_parsed = 0
    parse_failures: List[Tuple[str, str]] = []
    num_functions = 0
    num_ok = 0
    num_data_skipped = 0
    failure_counts: Counter[str] = Counter()

    for path in asm_files:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                asm_file = parse_file(f, arch, options)
            files_parsed += 1
        except Exception as e:
            parse_failures.append((path.name, str(e).splitlines()[0]))
            continue

        asm_data = AsmData()
        asm_file.asm_data.merge_into(asm_data)
        function_names = {fn.name for fn in asm_file.functions}
        typepool = TypePool(
            unknown_field_prefix="unk_",
            unk_inference=False,
            union_field_overrides={},
        )
        global_info = build_global_info(
            asm_data, arch, options, function_names, typemap, typepool
        )

        for function in asm_file.functions:
            if not any(isinstance(part, Instruction) for part in function.body):
                # A label in the .text section followed by data only (e.g. a
                # partially-labeled jump table exported by Ghidra): not a
                # function at all. Skip it rather than counting it as a
                # decompilation failure.
                num_data_skipped += 1
                continue
            num_functions += 1
            try:
                decompile_function(function, options, global_info, print_warnings=False)
                num_ok += 1
            except Exception as e:
                failure_counts[classify(e)] += 1

    print(f"Files: {files_parsed}/{len(asm_files)} parsed", end="")
    if parse_failures:
        print(f" ({len(parse_failures)} parse failures)")
        for name, msg in parse_failures[:10]:
            print(f"    {name}: {msg}")
    else:
        print()
    print(
        f"Functions: {num_ok}/{num_functions} decompiled without failure "
        f"({100 * num_ok / max(num_functions, 1):.1f}%)"
    )
    if num_data_skipped:
        print(f"Skipped {num_data_skipped} instruction-less data labels")
    print(f"\nTop {cli_args.top} failure buckets:")
    for label, count in failure_counts.most_common(cli_args.top):
        print(f"  {count:>6}  {label}")


if __name__ == "__main__":
    main()
