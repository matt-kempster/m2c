from typing import Optional
import unittest

from m2c.arch_arm import Cc, parse_suffix


def parse_cc(cc: str) -> Optional[Cc]:
    if not cc:
        return None
    if cc == "hs":
        return Cc.CS
    if cc == "lo":
        return Cc.CC
    return Cc(cc)


class TestArm(unittest.TestCase):
    def test_parse_suffix(self) -> None:
        """Test that parse_suffix handles all known instructions correctly.

        We check both UAL and legacy syntaxes, and test that there are no
        ambiguous instruction mnemonics. (In the past we have encountered bugs
        like parsing bls as bl + s instead of b + ls.)"""
        dirs = ",ia,ib,da,db,fa,fd,ea,ed".split(",")
        cc_strs = ",hs,lo,eq,ne,cs,cc,mi,pl,vs,vc,hi,ls,ge,lt,gt,le,al".split(",")
        ccs = [(s, parse_cc(s)) for s in cc_strs]

        for mn in (
            "b",
            "bl",
            "blx",
            "bx",
            "clz",
            "cmn",
            "cmp",
            "cpy",
            "ldr",
            "ldrb",
            "ldrh",
            "ldrsb",
            "ldrsh",
            "mla",
            "nop",
            "pop",
            "push",
            "rbit",
            "rev",
            "rev16",
            "sdiv",
            "sdiv",
            "smlabb",
            "smulbb",
            "str",
            "strb",
            "strh",
            "tablejmp.fictive",
            "teq",
            "tst",
            "udiv",
        ):
            for cc_str, cc in ccs:
                self.assertEqual(parse_suffix(mn + cc_str), (mn, cc, "", ""))

        for cc_str, cc in ccs:
            for sz in ("sb", "b", "sh", "h"):
                self.assertEqual(
                    parse_suffix(f"ldr{cc_str}{sz}"), (f"ldr{sz}", cc, "", "")
                )
            for sz in ("b", "h"):
                self.assertEqual(
                    parse_suffix(f"str{cc_str}{sz}"), (f"str{sz}", cc, "", "")
                )

        for mn in (
            "adc",
            "add",
            "and",
            "asr",
            "bic",
            "eor",
            "lsl",
            "lsr",
            "mov",
            "mul",
            "mvn",
            "neg",
            "orn",
            "orr",
            "ror",
            "rrx",
            "rsb",
            "rsc",
            "sbc",
            "smlal",
            "smull",
            "sub",
            "umlal",
            "umull",
        ):
            for cc_str, cc in ccs:
                for s in ("", "s"):
                    self.assertEqual(parse_suffix(mn + cc_str + s), (mn, cc, s, ""))
                    self.assertEqual(parse_suffix(mn + s + cc_str), (mn, cc, s, ""))

        for mn in ("stm", "ldm"):
            for cc_str, cc in ccs:
                tr = {
                    "fa": ["da", "ib"],
                    "ea": ["db", "ia"],
                    "fd": ["ia", "db"],
                    "ed": ["ib", "da"],
                }
                for d in dirs:
                    exp = d
                    if d in tr:
                        exp = tr[d][0 if mn == "ldm" else 1]
                    self.assertEqual(
                        parse_suffix(f"{mn}{cc_str}{d}"), (mn, cc, "", exp)
                    )

        for mn in ("it", "cbz", "cbnz"):
            self.assertEqual(parse_suffix(mn), (mn, None, "", ""))
