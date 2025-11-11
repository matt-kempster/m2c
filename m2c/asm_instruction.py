"""Functions and classes useful for parsing an arbitrary assembly instruction."""

from __future__ import annotations
import abc
from dataclasses import dataclass, field
from enum import Enum
import string
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

from .error import DecompFailure, static_assert_unreachable
from .options import Target


ARM_BARREL_SHIFTER_OPS = ("lsl", "lsr", "asr", "ror", "rrx")
OP_PRECEDENCE = {
    "*": 0,
    "+": 1,
    "-": 1,
    "<<": 2,
    ">>": 2,
    "&": 3,
    "^": 4,
    "|": 5,
}
MAX_PRECEDENCE = 6


@dataclass(frozen=True)
class Register:
    register_name: str

    def is_float(self) -> bool:
        name = self.register_name
        return bool(name) and name[0] == "f" and name != "fp"

    def arm_index(self) -> int:
        index = {"sp": 13, "lr": 14, "pc": 15}.get(self.register_name)
        if index is not None:
            return index
        assert self.register_name.startswith("r"), self.register_name
        return int(self.register_name[1:])

    @staticmethod
    def fictive(name: str, suffix: str = "") -> Register:
        return Register(f"{name}_fictive_{suffix}")

    def __str__(self) -> str:
        return f"${self.register_name}"


@dataclass(frozen=True)
class RegisterList:
    regs: List[Register]

    def __str__(self) -> str:
        return "{" + ",".join(str(r) for r in self.regs) + "}"


@dataclass(frozen=True)
class AsmGlobalSymbol:
    symbol_name: str

    def __str__(self) -> str:
        return self.symbol_name


@dataclass(frozen=True)
class AsmSectionGlobalSymbol(AsmGlobalSymbol):
    section_name: str
    addend: int


def asm_section_global_symbol(section_name: str, addend: int) -> AsmSectionGlobalSymbol:
    return AsmSectionGlobalSymbol(
        symbol_name=f"__{section_name}{hex(addend)[2:].upper()}",
        section_name=section_name,
        addend=addend,
    )


@dataclass(frozen=True)
class Macro:
    macro_name: str
    argument: Argument

    def __str__(self) -> str:
        return f"%{self.macro_name}({self.argument})"


@dataclass(frozen=True)
class AsmLiteral:
    value: int

    def as_s16(self) -> int:
        return ((self.value + 0x8000) & 0xFFFF) - 0x8000

    def __str__(self) -> str:
        return hex(self.value)


class Writeback(Enum):
    PRE = "pre"
    POST = "post"


@dataclass(frozen=True)
class AsmAddressMode:
    base: Register
    addend: Argument
    writeback: Optional[Writeback]

    def addend_as_literal(self) -> int:
        assert isinstance(self.addend, AsmLiteral)
        return self.addend.value

    def __str__(self) -> str:
        if self.writeback is not None:
            add_str = ", {self.addend}" if self.addend != AsmLiteral(0) else ""
            if self.writeback == Writeback.PRE:
                return f"[{self.base}{add_str}]!"
            else:
                return f"[{self.base}]{add_str}"
        if self.addend == AsmLiteral(0):
            return f"({self.base})"
        else:
            return f"{self.addend}({self.base})"


@dataclass(frozen=True)
class BinOp:
    op: str
    lhs: Argument
    rhs: Argument

    def __str__(self) -> str:
        return f"{self.lhs} {self.op} {self.rhs}"


@dataclass(frozen=True)
class JumpTarget:
    target: str

    def __str__(self) -> str:
        return self.target


Argument = Union[
    Register, RegisterList, AsmGlobalSymbol, AsmAddressMode, Macro, AsmLiteral, BinOp
]


@dataclass(frozen=True)
class AsmInstruction:
    mnemonic: str
    args: List[Argument]

    def __str__(self) -> str:
        if not self.args:
            return self.mnemonic
        args = ", ".join(str(arg) for arg in self.args)
        return f"{self.mnemonic} {args}"


class ArchAsmParsing(abc.ABC):
    """Arch-specific information needed to parse asm."""

    all_regs: List[Register]
    aliased_regs: Dict[str, Register]
    supports_dollar_regs: bool

    @abc.abstractmethod
    def normalize_instruction(
        self, instr: AsmInstruction, asm_state: AsmState
    ) -> AsmInstruction: ...

    def should_ignore_symbol(self, symbol: str) -> bool:
        """
        Allow architectures to ignore certain bare symbols during parsing.
        Used for syntactic sugar such as x86 size prefixes (e.g. `dword ptr`).
        """
        return False


class NaiveParsingArch(ArchAsmParsing):
    """A fake arch that can parse asm in a naive fashion. Used by the pattern matching
    machinery to reduce arch dependence."""

    all_regs: List[Register] = []
    aliased_regs: Dict[str, Register] = {}
    supports_dollar_regs = True

    def normalize_instruction(
        self, instr: AsmInstruction, asm_state: AsmState
    ) -> AsmInstruction:
        return instr


@dataclass
class RegFormatter:
    """Converts register names used in input assembly to the internal register representation,
    saves the input's names, and converts back to the input's names for the output."""

    used_names: Dict[Register, str] = field(default_factory=dict)
    aliases: Dict[Register, Set[str]] = field(default_factory=dict)

    def parse(self, reg_name: str, arch: ArchAsmParsing) -> Register:
        return arch.aliased_regs.get(reg_name, Register(reg_name))

    def parse_and_store(self, reg_name: str, arch: ArchAsmParsing) -> Register:
        reg_name = reg_name.lower()
        internal_reg = arch.aliased_regs.get(reg_name, Register(reg_name))
        self.aliases.setdefault(internal_reg, set()).add(reg_name)
        existing_reg_name = self.used_names.get(internal_reg)
        if existing_reg_name is None:
            self.used_names[internal_reg] = reg_name
        return internal_reg

    def format(self, reg: Register) -> str:
        return self.used_names.get(reg, reg.register_name)

    def aliases_for(self, reg: Register) -> Set[str]:
        default_name = self.used_names.get(reg, reg.register_name)
        return self.aliases.get(reg, {default_name})


@dataclass
class AsmState:
    # None means "explicitly undefined"
    defines: Dict[str, Optional[int]] = field(default_factory=dict)
    reg_formatter: RegFormatter = field(default_factory=RegFormatter)
    is_thumb: bool = False
    is_unified: bool = False


valid_word = string.ascii_letters + string.digits + "_$."
valid_number = "-xX" + string.hexdigits


def parse_word(elems: List[str], valid: str = valid_word) -> str:
    ret: str = ""
    while elems and elems[0] in valid:
        ret += elems.pop(0)
    return ret


def parse_quoted(elems: List[str], quote_char: str) -> str:
    ret: str = ""
    while elems and elems[0] != quote_char:
        # Handle backslash-escaped characters
        # We only need to care about \\, \" and \' in this context.
        if elems[0] == "\\":
            elems.pop(0)
            if not elems:
                break
        ret += elems.pop(0)
    return ret


def parse_number(number_str: str) -> int:
    if number_str[0] == "0":
        assert len(number_str) == 1 or number_str[1] in "xX"
    ret = int(number_str, 0)
    return ret


def constant_fold(arg: Argument, asm_state: AsmState) -> Argument:
    if isinstance(arg, AsmGlobalSymbol):
        value = asm_state.defines.get(arg.symbol_name)
        if value is not None:
            return AsmLiteral(value)
    if not isinstance(arg, BinOp):
        return arg
    lhs = constant_fold(arg.lhs, asm_state)
    rhs = constant_fold(arg.rhs, asm_state)
    if isinstance(lhs, AsmLiteral) and isinstance(rhs, AsmLiteral):
        if arg.op == "+":
            return AsmLiteral(lhs.value + rhs.value)
        if arg.op == "-":
            return AsmLiteral(lhs.value - rhs.value)
        if arg.op == "*":
            return AsmLiteral(lhs.value * rhs.value)
        if arg.op == ">>":
            return AsmLiteral(lhs.value >> rhs.value)
        if arg.op == "<<":
            return AsmLiteral(lhs.value << rhs.value)
        if arg.op == "&":
            return AsmLiteral(lhs.value & rhs.value)
        if arg.op == "|":
            return AsmLiteral(lhs.value | rhs.value)
        if arg.op == "^":
            return AsmLiteral(lhs.value ^ rhs.value)
    return BinOp(arg.op, lhs, rhs)


def replace_bare_reg(
    arg: Argument, arch: ArchAsmParsing, asm_state: AsmState
) -> Argument:
    """If `arg` is an AsmGlobalSymbol whose name matches a known or aliased register,
    convert it into a Register and return it. Otherwise, return the original `arg`."""
    if isinstance(arg, AsmGlobalSymbol):
        reg_name = arg.symbol_name.lower()
        if Register(reg_name) in arch.all_regs or reg_name in arch.aliased_regs:
            return asm_state.reg_formatter.parse_and_store(reg_name, arch)
    return arg


def get_jump_target(label: Argument) -> JumpTarget:
    assert isinstance(label, AsmGlobalSymbol), "invalid branch target"
    return JumpTarget(label.symbol_name)


# Main parser.
def parse_arg_elems(
    arg_elems: List[str],
    arch: ArchAsmParsing,
    asm_state: AsmState,
    *,
    top_level: bool,
    do_constant_fold: bool = True,
    do_replace_bare_reg: bool = True,
    precedence_cap: int = MAX_PRECEDENCE,
) -> Argument:
    value: Optional[Argument] = None
    is_arm_arch = getattr(arch, "arch", None) == Target.ArchEnum.ARM

    def consume_ws() -> None:
        while arg_elems and arg_elems[0].isspace():
            arg_elems.pop(0)

    def expect(n: str) -> str:
        assert arg_elems, f"Expected one of {list(n)}, but reached end of string"
        g = arg_elems.pop(0)
        assert g in n, f"Expected one of {list(n)}, got {g} (rest: {arg_elems})"
        return g

    while True:
        consume_ws()
        if not arg_elems:
            break
        tok: str = arg_elems[0]
        if tok == ",":
            break
        elif tok == "$" and arch.supports_dollar_regs:
            # Register.
            assert value is None
            word = parse_word(arg_elems)
            reg = word[1:]
            if "$" in reg:
                # If there is a second $ in the word, it's a symbol
                value = AsmGlobalSymbol(word)
            else:
                value = asm_state.reg_formatter.parse_and_store(reg, arch)
        elif tok == "#":
            # ARM immediate.
            assert value is None
            expect("#")
        elif tok == "{":
            # ARM register list.
            assert value is None
            arg_elems.pop(0)
            li: List[Register] = []
            while True:
                consume_ws()
                if li:
                    if expect(",}") == "}":
                        break
                    consume_ws()
                word = parse_word(arg_elems)
                reg1 = asm_state.reg_formatter.parse_and_store(word, arch)
                consume_ws()
                if arg_elems[0] != "-":
                    li.append(reg1)
                    continue
                arg_elems.pop(0)
                consume_ws()
                word = parse_word(arg_elems)
                reg2 = asm_state.reg_formatter.parse_and_store(word, arch)
                for i in range(reg1.arm_index(), reg2.arm_index() + 1):
                    li.append(asm_state.reg_formatter.parse(f"r{i}", arch))
            consume_ws()
            if arg_elems and arg_elems[0] == "^":
                expect("^")
            if len(li) > 1:
                li.sort(key=lambda r: r.arm_index())
            value = RegisterList(li)
        elif tok == ".":
            # Either a jump target (i.e. a label), or a section reference.
            assert value is None
            arg_elems.pop(0)
            word = parse_word(arg_elems)
            if word in ["data", "sdata", "rodata", "rdata", "bss", "sbss", "text"]:
                value = asm_section_global_symbol(word, 0)
            else:
                value = AsmGlobalSymbol("." + word)
        elif tok == "%":
            # A MIPS reloc macro, e.g. %hi(...) or %lo(...).
            assert value is None
            arg_elems.pop(0)
            macro_name = parse_word(arg_elems)
            assert macro_name
            expect("(")
            # Get the argument of the macro (which must exist).
            m = parse_arg_elems(
                arg_elems,
                arch,
                asm_state,
                top_level=False,
                do_replace_bare_reg=False,
            )
            expect(")")
            # A macro may be the lhs of an AsmAddressMode, so we don't return here.
            value = Macro(macro_name, m)
        elif tok.lower() == "o" and "".join(arg_elems[:6]).lower().startswith("offset"):
            # MASM-style "offset symbol"
            for _ in range(6):
                arg_elems.pop(0)
            consume_ws()
            symbol = parse_word(arg_elems)
            value = AsmGlobalSymbol(symbol)
        elif tok in (")", "]"):
            # Break out to the parent of this call, since we are in parens.
            break
        elif tok in string.digits:
            # Try a number.
            assert value is None
            word = parse_word(arg_elems, valid_word)
            value = AsmLiteral(parse_number(word))
        elif tok == "-" and value is None:
            # Negated number, or ARM negated register.
            expect("-")
            consume_ws()
            word = parse_word(arg_elems, valid_word)
            assert word
            if word[0] in valid_number:
                value = AsmLiteral(-parse_number(word))
            else:
                val = replace_bare_reg(AsmGlobalSymbol(word), arch, asm_state)
                value = BinOp("-", AsmLiteral(0), val)
        elif tok == "(":
            if value is not None and not top_level:
                # Only allow parsing AsmAddressMode at top level. This makes us parse
                # a+b(c) as (a+b)(c) instead of a+(b(c)).
                break
            # Address mode or binary operation.
            expect("(")
            # Get what is being dereferenced.
            rhs = parse_arg_elems(
                arg_elems,
                arch,
                asm_state,
                top_level=False,
                do_constant_fold=False,
            )
            expect(")")
            if isinstance(rhs, BinOp):
                # Binary operation.
                assert value is None
                value = constant_fold(rhs, asm_state)
            else:
                # Address mode.
                if rhs == AsmLiteral(0):
                    rhs = Register("zero")
                if isinstance(rhs, AsmGlobalSymbol):
                    # Allow absolute addresses inside brackets/parentheses,
                    # e.g. MOV EAX, [_DAT_XXXX].
                    assert top_level
                    assert value is None
                    value = AsmAddressMode(Register("zero"), rhs, None)
                else:
                    assert top_level
                    assert isinstance(rhs, Register)
                    value = constant_fold(value or AsmLiteral(0), asm_state)
                    value = AsmAddressMode(rhs, value, None)
        elif tok == "[":
            # ARM address mode
            assert value is None
            expect("[")
            val = parse_arg_elems(arg_elems, arch, asm_state, top_level=False)
            initial_addend: Optional[Argument] = None
            scale = AsmLiteral(1)
            index: Optional[Register] = None
            base_candidate = replace_bare_reg(val, arch, asm_state)
            if isinstance(base_candidate, Register):
                val = base_candidate
            elif isinstance(val, BinOp):
                def scaled_term(term: Argument) -> Optional[Tuple[Register, AsmLiteral]]:
                    if (
                        isinstance(term, BinOp)
                        and term.op == "*"
                        and isinstance(term.lhs, Register)
                        and isinstance(term.rhs, AsmLiteral)
                    ):
                        return term.lhs, term.rhs
                    return None

                if val.op in ("+", "-"):
                    op_sign = 1 if val.op == "+" else -1
                    left = replace_bare_reg(val.lhs, arch, asm_state)
                    right = replace_bare_reg(val.rhs, arch, asm_state)
                    base_candidate = left if isinstance(left, Register) else None
                    if isinstance(base_candidate, Register):
                        val = base_candidate
                    else:
                        scaled = scaled_term(left)
                        if scaled is not None:
                            index, scale = scaled
                            val = Register("zero")
                    scaled = scaled_term(right)
                    if scaled is not None:
                        index, scale = scaled
                        if not isinstance(base_candidate, Register):
                            val = Register("zero")
                    elif isinstance(right, AsmLiteral):
                        literal = op_sign * right.value
                        initial_addend = AsmLiteral(literal)
                elif val.op == "*":
                    if isinstance(val.lhs, Register) and isinstance(val.rhs, AsmLiteral):
                        index = val.lhs
                        scale = val.rhs
                        val = Register("zero")
                else:
                    scaled = scaled_term(val)
                    if scaled is not None:
                        index, scale = scaled
                        val = Register("zero")
            elif isinstance(val, AsmGlobalSymbol):
                initial_addend = val
                val = Register("zero")
            addend: Optional[Argument] = None
            if initial_addend is not None:
                addend = initial_addend
            if not isinstance(val, Register):
                extra = constant_fold(val, asm_state)
                if addend is None:
                    addend = extra
                else:
                    addend = constant_fold(BinOp("+", addend, extra), asm_state)
                val = Register("zero")
            assert isinstance(val, Register)
            if expect(",]") == ",":
                addend = parse_arg_elems(arg_elems, arch, asm_state, top_level=False)
                if expect(",]") == ",":
                    consume_ws()
                    op = parse_word(arg_elems).lower()
                    assert op in ARM_BARREL_SHIFTER_OPS
                    shift: Argument
                    if op == "rrx":
                        shift = AsmLiteral(1)
                    else:
                        shift = parse_arg_elems(
                            arg_elems, arch, asm_state, top_level=False
                        )
                    addend = BinOp(op, addend, constant_fold(shift, asm_state))
                    expect("]")
            if index is not None:
                scaled = BinaryOp("*", index, scale)
                scaled = constant_fold(scaled, asm_state)
                if addend is None:
                    addend = scaled
                else:
                    addend = constant_fold(BinOp("+", addend, scaled), asm_state)
            consume_ws()
            writeback: Optional[Writeback] = None
            if arg_elems and arg_elems[0] == "!":
                expect("!")
                writeback = Writeback.PRE
            elif arg_elems and arg_elems[0] == "," and is_arm_arch:
                expect(",")
                assert addend is None
                addend = parse_arg_elems(arg_elems, arch, asm_state, top_level=False)
                writeback = Writeback.POST
            if addend is None:
                addend = AsmLiteral(0)
            else:
                def replace_regs(arg: Argument) -> Argument:
                    if isinstance(arg, AsmGlobalSymbol):
                        return replace_bare_reg(arg, arch, asm_state)
                    if isinstance(arg, BinOp):
                        return BinOp(
                            arg.op,
                            replace_regs(arg.lhs),
                            replace_regs(arg.rhs),
                        )
                    return arg

                addend = replace_regs(addend)
            value = AsmAddressMode(val, addend, writeback)
        elif tok == "!":
            # ARM writeback indicator, e.g. "ldmia sp!, {r3, r4, r5, pc}".
            # Let's abuse AsmAddressMode for this.
            expect("!")
            if value is not None:
                value = replace_bare_reg(value, arch, asm_state)
            assert isinstance(value, Register)
            value = AsmAddressMode(value, AsmLiteral(0), Writeback.PRE)
        elif tok == '"':
            # Quoted global symbol.
            expect('"')
            assert value is None
            word = parse_quoted(arg_elems, '"')
            value = AsmGlobalSymbol(word)
            expect('"')
        elif tok in valid_word:
            # Global symbol.
            assert value is None
            word = parse_word(arg_elems)
            if arch.should_ignore_symbol(word.lower()):
                continue
            value = AsmGlobalSymbol(word)

            op = word.lower()
            if top_level and op in ARM_BARREL_SHIFTER_OPS:
                consume_ws()
                if arg_elems and arg_elems[0] not in ",)@":
                    # ARM barrel shifter operation. This should be folded into
                    # the previous comma-separated operation; the caller will
                    # do that for us.
                    shift = parse_arg_elems(arg_elems, arch, asm_state, top_level=False)
                    value = BinOp(op, AsmLiteral(0), shift)
                    assert not arg_elems
                elif op == "rrx":
                    value = BinOp(op, AsmLiteral(0), AsmLiteral(1))
                    assert not arg_elems
        elif tok in "<>+-*&|^":
            # Binary operators, used e.g. to modify global symbols or constants.
            assert isinstance(value, (AsmLiteral, AsmGlobalSymbol, BinOp))

            if tok in "<>":
                # bitshifts
                op = tok + tok
            else:
                op = tok

            precedence = OP_PRECEDENCE[op]
            if precedence >= precedence_cap:
                break

            for c in op:
                expect(c)

            if op == "-" and arg_elems[0] == "_":
                # Parse `sym-_SDA_BASE_` as a Macro, equivalently to `sym@sda21`
                reloc_name = parse_word(arg_elems)
                if reloc_name not in ("_SDA_BASE_", "_SDA2_BASE_"):
                    raise DecompFailure(
                        f"Unexpected symbol {reloc_name} in subtraction expression"
                    )
                value = Macro("sda21", value)
            else:
                rhs = parse_arg_elems(
                    arg_elems,
                    arch,
                    asm_state,
                    top_level=False,
                    do_replace_bare_reg=False,
                    precedence_cap=precedence,
                )
                if (
                    op == "+"
                    and isinstance(rhs, AsmLiteral)
                    and isinstance(value, AsmSectionGlobalSymbol)
                ):
                    value = asm_section_global_symbol(
                        value.section_name, value.addend + rhs.value
                    )
                else:
                    value = BinOp(op, value, rhs)
        elif tok == "@":
            # A relocation (e.g. (...)@ha or (...)@l).
            if not top_level:
                # Parse a+b@l as (a+b)@l, not a+(b@l)
                break
            arg_elems.pop(0)
            reloc_name = parse_word(arg_elems)
            assert reloc_name in ("h", "ha", "l", "sda2", "sda21")
            assert value
            value = Macro(reloc_name, value)
        else:
            assert False, f"Unknown token {tok} in {arg_elems}"

    assert value is not None
    if do_constant_fold:
        value = constant_fold(value, asm_state)
    if do_replace_bare_reg:
        value = replace_bare_reg(value, arch, asm_state)
    return value


def parse_args(
    args: str,
    arch: ArchAsmParsing,
    asm_state: AsmState,
) -> List[Argument]:
    arg_elems: List[str] = list(args.strip())
    output: List[Argument] = []
    while arg_elems:
        ret = parse_arg_elems(arg_elems, arch, asm_state, top_level=True)
        if isinstance(ret, BinOp) and ret.op in ARM_BARREL_SHIFTER_OPS:
            assert output
            output[-1] = BinOp(ret.op, output[-1], ret.rhs)
            continue
        output.append(ret)
        if arg_elems:
            comma = arg_elems.pop(0)
            assert comma == ","
    return output


def parse_arg(arg: str, arch: ArchAsmParsing, asm_state: AsmState) -> Argument:
    chars = list(arg)
    value = parse_arg_elems(
        chars, arch, asm_state, top_level=False, do_replace_bare_reg=False
    )
    if chars:
        raise Exception(f"Failed to parse value: {arg}")
    return value


def parse_asm_instruction(
    line: str,
    arch: ArchAsmParsing,
    asm_state: AsmState,
) -> AsmInstruction:
    # First token is instruction name, rest is args.
    line = line.strip()
    mnemonic, _, args_str = line.partition(" ")
    # Parse arguments.
    args = parse_args(args_str, arch, asm_state)
    instr = AsmInstruction(mnemonic.lower(), args)
    return arch.normalize_instruction(instr, asm_state)


def traverse_arg(arg: Argument) -> Iterator[Argument]:
    yield arg
    if isinstance(arg, (Register, AsmLiteral, AsmGlobalSymbol)):
        pass
    elif isinstance(arg, RegisterList):
        for reg in arg.regs:
            yield reg
    elif isinstance(arg, AsmAddressMode):
        yield arg.base
        yield from traverse_arg(arg.addend)
    elif isinstance(arg, Macro):
        yield from traverse_arg(arg.argument)
    elif isinstance(arg, BinOp):
        yield from traverse_arg(arg.lhs)
        yield from traverse_arg(arg.rhs)
    else:
        static_assert_unreachable(arg)
