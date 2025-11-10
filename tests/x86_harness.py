#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m2c.arch_x86 import X86Arch
from m2c.asm_file import parse_file
from m2c.error import DecompFailure
from m2c.main import parse_flags

DEFAULT_ASM_ROOT = Path("/Users/marijn/Projects/legoland/port2/project/asm")
UNSUPPORTED_MARKER = "x86 unsupported instruction:"


def classify_failure(exc: DecompFailure) -> Tuple[str, str]:
    msg = str(exc).strip()
    if UNSUPPORTED_MARKER in msg:
        detail = msg.split(UNSUPPORTED_MARKER, 1)[1].strip()
        mnemonic = detail.split(" ", 1)[0]
        return "unsupported", mnemonic
    return "other", msg.splitlines()[0]


def run_harness(
    asm_root: Path, limit: Optional[int], sample_limit: int, top_n: int
) -> None:
    asm_files = sorted(asm_root.glob("*.asm"))
    if not asm_files:
        print(f"No .asm files found under {asm_root}")
        return
    if limit is not None:
        asm_files = asm_files[:limit]

    arch = X86Arch()
    unsupported_counts: Counter[str] = Counter()
    samples: Dict[str, List[str]] = defaultdict(list)
    other_errors: List[Tuple[str, str]] = []

    for path in asm_files:
        options = parse_flags(["-t", "x86-gcc-c", "--stop-on-error", str(path)])
        try:
            with open(path, "r", encoding="utf-8-sig") as handle:
                parse_file(handle, arch, options)
        except DecompFailure as exc:
            kind, label = classify_failure(exc)
            if kind == "unsupported":
                unsupported_counts[label] += 1
                if len(samples[label]) < sample_limit:
                    samples[label].append(path.name)
            else:
                other_errors.append((path.name, label))

    total = len(asm_files)
    failed = sum(unsupported_counts.values()) + len(other_errors)
    print(f"Scanned {total} asm files ({failed} with errors).")

    if unsupported_counts:
        print("\nTop unsupported instructions:")
        for mnemonic, count in unsupported_counts.most_common(top_n):
            sample_str = ", ".join(samples[mnemonic])
            sample_suffix = f" [{sample_str}]" if sample_str else ""
            print(f"  {mnemonic:<12} {count:>5} hits{sample_suffix}")
    else:
        print("\nNo unsupported instructions detected.")

    if other_errors:
        print(f"\nOther failures ({len(other_errors)}):")
        for filename, message in other_errors:
            print(f"  {filename}: {message}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run m2c on many x86 assembly files and summarize unsupported instructions."
    )
    parser.add_argument(
        "--asm-root",
        type=Path,
        default=DEFAULT_ASM_ROOT,
        help=f"Directory containing .asm files (default: {DEFAULT_ASM_ROOT})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only scan this many asm files (useful for quick iterations).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Number of sample filenames to store per unsupported instruction.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="How many instruction groups to display in the summary.",
    )
    args = parser.parse_args()

    run_harness(args.asm_root, args.limit, args.sample_limit, args.top)


if __name__ == "__main__":
    main()
