"""Functions and classes useful for parsing an arbitrary MIPS instruction.
"""
import ast
import re
import string
import sys
import typing
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import attr


@attr.s(frozen=True)
class Register:
    register_name: str = attr.ib()

    def is_callee_save(self) -> bool:
        return bool(re.match("s[0-7]", self.register_name))

    def is_float(self) -> bool:
        return self.register_name[0] == "f" and self.register_name != "fp"

    def __str__(self) -> str:
        return f"${self.register_name}"


@attr.s(frozen=True)
class AsmGlobalSymbol:
    symbol_name: str = attr.ib()

    def __str__(self) -> str:
        return self.symbol_name


@attr.s(frozen=True)
class Macro:
    macro_name: str = attr.ib()
    argument: "Argument" = attr.ib()  # forward-declare

    def __str__(self) -> str:
        return f"%{self.macro_name}({self.argument})"


@attr.s(frozen=True)
class AsmLiteral:
    value: int = attr.ib()

    def __str__(self) -> str:
        return hex(self.value)


@attr.s(frozen=True)
class AsmAddressMode:
    lhs: Union[AsmLiteral, Macro, None] = attr.ib()
    rhs: Register = attr.ib()

    def __str__(self) -> str:
        if self.lhs is not None:
            return f"{self.lhs}({self.rhs})"
        else:
            return f"({self.rhs})"


@attr.s(frozen=True)
class BinOp:
    op: str = attr.ib()
    lhs: "Argument" = attr.ib()
    rhs: "Argument" = attr.ib()

    def __str__(self) -> str:
        return f"{self.lhs} {self.op} {self.rhs}"


@attr.s(frozen=True)
class JumpTarget:
    target: str = attr.ib()

    def __str__(self) -> str:
        return f".{self.target}"


Argument = Union[
    Register, AsmGlobalSymbol, AsmAddressMode, Macro, AsmLiteral, BinOp, JumpTarget
]

valid_word = string.ascii_letters + string.digits + "_"
valid_number = "-x" + string.hexdigits


def parse_word(elems: List[str], valid: str = valid_word) -> str:
    S: str = ""
    while elems and elems[0] in valid:
        S += elems.pop(0)
    return S


def parse_number(elems: List[str]) -> int:
    number_str = parse_word(elems, valid_number)
    ret = ast.literal_eval(number_str)
    assert isinstance(ret, int)
    return ret


# Main parser.
def parse_arg_elems(arg_elems: List[str]) -> Optional[Argument]:
    value: Optional[Argument] = None

    def expect(n: str) -> str:
        g = arg_elems.pop(0)
        assert g in n, f"Expected one of {list(n)}, got {g} (rest: {arg_elems})"
        return g

    while arg_elems:
        tok: str = arg_elems[0]
        if tok.isspace():
            # Ignore whitespace.
            arg_elems.pop(0)
        elif tok == "$":
            # Register.
            assert value is None
            arg_elems.pop(0)
            reg = parse_word(arg_elems)
            if reg == "s8":
                reg = "fp"
            value = Register(reg)
        elif tok == ".":
            # A jump target (i.e. a label).
            assert value is None
            arg_elems.pop(0)
            value = JumpTarget(parse_word(arg_elems))
        elif tok == "%":
            # A macro (i.e. %hi(...) or %lo(...)).
            assert value is None
            arg_elems.pop(0)
            macro_name = parse_word(arg_elems)
            assert macro_name in ("hi", "lo")
            expect("(")
            # Get the argument of the macro (which must exist).
            m = parse_arg_elems(arg_elems)
            assert m is not None
            expect(")")
            # A macro may be the lhs of an AsmAddressMode, so we don't return here.
            value = Macro(macro_name, m)
        elif tok == ")":
            # Break out to the parent of this call, since we are in parens.
            break
        elif tok in string.digits or (tok == "-" and value is None):
            # Try a number.
            assert value is None
            value = AsmLiteral(parse_number(arg_elems))
        elif tok == "(":
            # Address mode or binary operation.
            # There was possibly an offset, so value could be a AsmLiteral or Macro.
            assert value is None or isinstance(value, (AsmLiteral, Macro))
            expect("(")
            # Get what is being dereferenced.
            rhs = parse_arg_elems(arg_elems)
            assert rhs is not None
            expect(")")
            if isinstance(rhs, BinOp):
                # Binary operation.
                return rhs
            else:
                # Address mode.
                assert isinstance(rhs, Register)
                value = AsmAddressMode(value, rhs)
        elif tok in valid_word:
            # Global symbol.
            assert value is None
            value = AsmGlobalSymbol(parse_word(arg_elems))
        elif tok in ">+-&":
            # Binary operators, used e.g. to modify global symbols or constants.
            assert isinstance(value, (AsmLiteral, AsmGlobalSymbol))

            if tok == ">":
                expect(">")
                expect(">")
                op = ">>"
            else:
                op = expect("&+-")

            rhs = parse_arg_elems(arg_elems)
            # These operators can only use constants as the right-hand-side.
            assert isinstance(rhs, AsmLiteral)
            return BinOp(op, value, rhs)
        else:
            assert False, f"Unknown token {tok} in {arg_elems}"

    return value


def parse_arg(arg: str) -> Optional[Argument]:
    arg_elems: List[str] = list(arg)
    return parse_arg_elems(arg_elems)


@attr.s(frozen=True)
class Instruction:
    mnemonic: str = attr.ib()
    args: List[Argument] = attr.ib()
    emit_goto: bool = attr.ib(default=False)

    def is_branch_instruction(self) -> bool:
        return (
            self.mnemonic
            in [
                "b",
                "beq",
                "bne",
                "beqz",
                "bnez",
                "bgez",
                "bgtz",
                "blez",
                "bltz",
                "bc1t",
                "bc1f",
            ]
            or self.is_branch_likely_instruction()
        )

    def is_branch_likely_instruction(self) -> bool:
        return self.mnemonic in [
            "beql",
            "bnel",
            "beqzl",
            "bnezl",
            "bgezl",
            "bgtzl",
            "blezl",
            "bltzl",
            "bc1tl",
            "bc1fl",
        ]

    def get_branch_target(self) -> JumpTarget:
        label = self.args[-1]
        assert isinstance(label, JumpTarget)
        return label

    def is_jump_instruction(self) -> bool:
        # (we don't treat jal/jalr as jumps, since control flow will return
        # after the call)
        return self.is_branch_instruction() or self.mnemonic in ["j", "jr"]

    def is_delay_slot_instruction(self) -> bool:
        return self.is_branch_instruction() or self.mnemonic in [
            "j",
            "jr",
            "jal",
            "jalr",
        ]

    def __str__(self) -> str:
        return f'    {self.mnemonic} {", ".join(str(arg) for arg in self.args)}'


def normalize_instruction(instr: Instruction) -> Instruction:
    args = instr.args
    if len(args) == 3:
        if instr.mnemonic == "or" and args[2] == Register("zero"):
            return Instruction("move", args[:2], instr.emit_goto)
        if instr.mnemonic == "nor" and args[1] == Register("zero"):
            return Instruction("not", [args[0], args[2]])
        if instr.mnemonic == "nor" and args[2] == Register("zero"):
            return Instruction("not", [args[0], args[1]])
        if instr.mnemonic == "addiu" and args[2] == AsmLiteral(0):
            return Instruction("move", args[:2], instr.emit_goto)
        if (
            instr.mnemonic == "ori"
            and args[1] == Register("zero")
            and isinstance(args[2], AsmLiteral)
        ):
            lit = AsmLiteral(args[2].value & 0xFFFF)
            return Instruction("li", [args[0], lit], instr.emit_goto)
        if (
            instr.mnemonic == "addiu"
            and args[1] == Register("zero")
            and isinstance(args[2], AsmLiteral)
        ):
            lit = AsmLiteral(((args[2].value + 0x8000) & 0xFFFF) - 0x8000)
            return Instruction("li", [args[0], lit], instr.emit_goto)
        if instr.mnemonic in ["bne", "beq", "beql", "bnel"] and args[1] == Register(
            "zero"
        ):
            mn = instr.mnemonic[:3] + "z" + instr.mnemonic[3:]
            return Instruction(mn, [args[0], args[2]], instr.emit_goto)
    return instr


def parse_instruction(line: str, emit_goto: bool) -> Instruction:
    try:
        # First token is instruction name, rest is args.
        line = line.strip()
        mnemonic, _, args_str = line.partition(" ")
        # Parse arguments.
        args: List[Argument] = list(
            filter(
                None, [parse_arg(arg_str.strip()) for arg_str in args_str.split(",")]
            )
        )
        instr = Instruction(mnemonic, args, emit_goto)
        return normalize_instruction(instr)
    except Exception as e:
        print(f"Failed to parse instruction: {line}\n", file=sys.stderr)
        raise e
