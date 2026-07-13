"""Pure preprocessing helpers for Intel-syntax assembly."""

from __future__ import annotations

import re
from typing import Dict, Tuple


PTR_WIDTHS: Dict[str, int] = {"byte": 1, "word": 2, "dword": 4, "qword": 8}
WIDTH_SUFFIXES: Dict[int, str] = {1: ".b", 2: ".w", 4: "", 8: ".q"}

RE_PTR = re.compile(r"\b(byte|word|dword|qword)\s+ptr\s+", re.IGNORECASE)
RE_OFFSET = re.compile(r"\boffset\s+", re.IGNORECASE)
RE_DISTANCE = re.compile(r"\b(short|near\s+ptr|far\s+ptr)\s+", re.IGNORECASE)
RE_ST_REG = re.compile(r"\bst\((\d)\)", re.IGNORECASE)
RE_SEGMENT = re.compile(r"\b([cdefgs]s):", re.IGNORECASE)

STRING_OP_MNEMONICS = {
    f"{op}{width}" for op in ("movs", "stos", "scas", "lods", "cmps") for width in "bwd"
}


def preprocess_intel_instruction(mnemonic: str, args: str) -> Tuple[str, str]:
    """Normalize Intel-only spelling before generic operand parsing."""
    mn = mnemonic
    if mn == "retn":
        mnemonic = mn = "ret"
    if mn in ("rep", "repe", "repne", "repz", "repnz"):
        parts = args.split(None, 1)
        if parts and parts[0].lower() in STRING_OP_MNEMONICS:
            return mnemonic, parts[0].lower()
    elif mn in STRING_OP_MNEMONICS and "es:" in args.lower():
        return mnemonic, ""

    widths = [PTR_WIDTHS[m.lower()] for m in RE_PTR.findall(args)]
    args = RE_PTR.sub("", args)
    args = RE_OFFSET.sub("", args)
    args = RE_DISTANCE.sub("", args)
    args = RE_ST_REG.sub(lambda m: f"st{m.group(1)}", args)
    segments = [m.lower() for m in RE_SEGMENT.findall(args)]
    args = RE_SEGMENT.sub("", args)
    for seg in segments:
        if seg in ("fs", "gs"):
            mnemonic += f".{seg}"
    if widths:
        mnemonic += WIDTH_SUFFIXES[min(widths)]
    return mnemonic, args
