from __future__ import annotations

from enum import Enum
from typing import Dict, Optional, Set, Tuple

from ..asm_instruction import Register


SUFFIXABLE_INSTRUCTIONS: Set[str] = {
    "adc",
    "add",
    "and",
    "asr",
    "b",
    "bic",
    "bl",
    "blx",
    "bx",
    "cbnz",
    "cbz",
    "clz",
    "cmn",
    "cmp",
    "cpy",
    "eor",
    "ldm",
    "ldr",
    "ldrb",
    "ldrh",
    "ldrsb",
    "ldrsh",
    "lsl",
    "lsr",
    "mla",
    "mls",
    "mov",
    "mul",
    "mvn",
    "neg",
    "nop",
    "orn",
    "orr",
    "pop",
    "push",
    "rbit",
    "rev",
    "rev16",
    "revsh",
    "ror",
    "rrx",
    "rsb",
    "rsc",
    "sbc",
    "sdiv",
    "smlabb",
    "smlal",
    "smmla",
    "smmls",
    "smmul",
    "smulbb",
    "smull",
    "stm",
    "str",
    "strb",
    "strh",
    "sub",
    "tablejmp.fictive",
    "teq",
    "tst",
    "udiv",
    "umlal",
    "umull",
}

LENGTH_THREE: Set[str] = {
    "adc",
    "add",
    "and",
    "asr",
    "bic",
    "eor",
    "lsl",
    "lsr",
    "mul",
    "orn",
    "orr",
    "ror",
    "rsb",
    "rsc",
    "sbc",
    "sub",
}

THUMB1_FLAG_SETTING: Set[str] = {
    "adc",
    "add",
    "and",
    "asr",
    "bic",
    "eor",
    "lsl",
    "lsr",
    "mov",
    "mvn",
    "orr",
    "rsb",
    "sub",
    "mul",
}

HI_REGS: Set[Register] = {
    Register("r8"),
    Register("r9"),
    Register("r10"),
    Register("r11"),
    Register("r12"),
    Register("sp"),
    Register("lr"),
    Register("pc"),
}


class Cc(Enum):
    EQ = "eq"
    NE = "ne"
    CS = "cs"
    CC = "cc"
    MI = "mi"
    PL = "pl"
    VS = "vs"
    VC = "vc"
    HI = "hi"
    LS = "ls"
    GE = "ge"
    LT = "lt"
    GT = "gt"
    LE = "le"
    AL = "al"


CC_REGS: Dict[Cc, Register] = {
    Cc.EQ: Register("z"),
    Cc.CS: Register("c"),
    Cc.MI: Register("n"),
    Cc.VS: Register("v"),
    Cc.HI: Register("hi"),
    Cc.GE: Register("ge"),
    Cc.GT: Register("gt"),
}


def negate_cond(cc: Cc) -> Cc:
    return {
        Cc.EQ: Cc.NE,
        Cc.NE: Cc.EQ,
        Cc.CS: Cc.CC,
        Cc.CC: Cc.CS,
        Cc.MI: Cc.PL,
        Cc.PL: Cc.MI,
        Cc.VS: Cc.VC,
        Cc.VC: Cc.VS,
        Cc.HI: Cc.LS,
        Cc.LS: Cc.HI,
        Cc.GE: Cc.LT,
        Cc.LT: Cc.GE,
        Cc.GT: Cc.LE,
        Cc.LE: Cc.GT,
    }[cc]


def factor_cond(cc: Cc) -> Tuple[Cc, bool]:
    return {
        Cc.EQ: (Cc.EQ, False),
        Cc.NE: (Cc.EQ, True),
        Cc.CS: (Cc.CS, False),
        Cc.CC: (Cc.CS, True),
        Cc.MI: (Cc.MI, False),
        Cc.PL: (Cc.MI, True),
        Cc.VS: (Cc.VS, False),
        Cc.VC: (Cc.VS, True),
        Cc.HI: (Cc.HI, False),
        Cc.LS: (Cc.HI, True),
        Cc.GE: (Cc.GE, False),
        Cc.LT: (Cc.GE, True),
        Cc.GT: (Cc.GT, False),
        Cc.LE: (Cc.GT, True),
    }[cc]


def parse_suffix(mnemonic: str) -> Tuple[str, Optional[Cc], str, str]:
    ldm = mnemonic.startswith("ldm")
    stm = mnemonic.startswith("stm")

    def strip_cc(mn: str) -> Tuple[str, Optional[Cc]]:
        for suffix in [cond.value for cond in Cc] + ["hs", "lo"]:
            if mn.endswith(suffix):
                if suffix == "hs":
                    cc = Cc.CS
                elif suffix == "lo":
                    cc = Cc.CC
                else:
                    cc = Cc(suffix)
                return mn[: -len(suffix)], cc
        return mn, None

    def strip_dir(mn: str) -> Tuple[str, str]:
        if not ldm and not stm:
            return mn, ""
        if any(mn.endswith(suffix) for suffix in ("ia", "ib", "da", "db")):
            return mn[:-2], mn[-2:]
        if any(mn.endswith(suffix) for suffix in ("fa", "ea", "fd", "ed")):
            # Pre-UAL syntax
            tr = {
                "fa": ["da", "ib"],
                "ea": ["db", "ia"],
                "fd": ["ia", "db"],
                "ed": ["ib", "da"],
            }
            return mn[:-2], tr[mn[-2:]][0 if ldm else 1]
        return mn, ""

    # bls should be parsed as b + ls, not bl + s
    s_ok = not mnemonic.startswith("b") or mnemonic.startswith("bic")

    # Strip memory size from the end (legacy ARM syntax). We re-attach it
    # later and treat it as part of the mnemonic.
    memsize = ""
    if mnemonic.startswith("str") or mnemonic.startswith("ldr"):
        # ldrhs should be parsed as ldr + hs, not ldrh + s
        s_ok = False
        for suffix in ("b", "h", "d"):
            if mnemonic.endswith(suffix):
                mnemonic = mnemonic[:-1]
                memsize = suffix
                break
        if (
            memsize in ("b", "h")
            and mnemonic.endswith("s")
            and (not strip_cc(mnemonic)[1] or strip_cc(mnemonic[:-1])[1])
        ):
            mnemonic = mnemonic[:-1]
            memsize = "s" + memsize

    orig_mn = mnemonic
    for cc_is_last in (False, True):
        mnemonic, direction = strip_dir(orig_mn)
        cc: Optional[Cc] = None
        set_flags = ""
        if cc_is_last:
            mnemonic, cc = strip_cc(mnemonic)
            if not direction:
                mnemonic, direction = strip_dir(mnemonic)
        if mnemonic in SUFFIXABLE_INSTRUCTIONS:
            return mnemonic + memsize, cc, set_flags, direction
        if mnemonic.endswith("s") and s_ok:
            set_flags = "s"
            mnemonic = mnemonic[:-1]
            if mnemonic in SUFFIXABLE_INSTRUCTIONS:
                return mnemonic + memsize, cc, set_flags, direction
        if not cc_is_last:
            mnemonic, cc = strip_cc(mnemonic)
            if mnemonic in SUFFIXABLE_INSTRUCTIONS:
                return mnemonic + memsize, cc, set_flags, direction

    return orig_mn + memsize, None, "", ""


def get_ldm_stm_offset(i: int, num_regs: int, direction: str) -> int:
    if direction == "ia":
        return 4 * i
    if direction == "ib":
        return 4 * (i + 1)
    if direction == "da":
        return 4 * (i + 1 - num_regs)
    if direction == "db":
        return 4 * (i - num_regs)
    raise AssertionError(f"bad ldm/stm direction {direction}")


def other_f64_reg(reg: Register) -> Register:
    num = int(reg.register_name[1:])
    return Register(f"r{num ^ 1}")


__all__ = [
    "CC_REGS",
    "Cc",
    "HI_REGS",
    "LENGTH_THREE",
    "SUFFIXABLE_INSTRUCTIONS",
    "THUMB1_FLAG_SETTING",
    "factor_cond",
    "get_ldm_stm_offset",
    "negate_cond",
    "other_f64_reg",
    "parse_suffix",
]
