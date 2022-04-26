"""Functions and classes useful for parsing an arbitrary MIPS instruction.
"""
import abc
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import string
from typing import Dict, Iterator, List, Optional, Set, Union

from .error import DecompFailure
from .options import Target


@dataclass(frozen=True)
class Register:
    register_name: str

    def is_float(self) -> bool:
        name = self.register_name
        return bool(name) and name[0] == "f" and name != "fp"

    def other_f64_reg(self) -> "Register":
        assert (
            self.is_float()
        ), "tried to get complement reg of non-floating point register"
        num = int(self.register_name[1:])
        return Register(f"f{num ^ 1}")

    def __str__(self) -> str:
        return f"${self.register_name}"


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
    argument: "Argument"

    def __str__(self) -> str:
        return f"%{self.macro_name}({self.argument})"


@dataclass(frozen=True)
class AsmLiteral:
    value: int

    def signed_value(self) -> int:
        return ((self.value + 0x8000) & 0xFFFF) - 0x8000

    def __str__(self) -> str:
        return hex(self.value)


@dataclass(frozen=True)
class AsmAddressMode:
    lhs: "Argument"
    rhs: Register

    def lhs_as_literal(self) -> int:
        assert isinstance(self.lhs, AsmLiteral)
        return self.lhs.signed_value()

    def __str__(self) -> str:
        if self.lhs == AsmLiteral(0):
            return f"({self.rhs})"
        else:
            return f"{self.lhs}({self.rhs})"


@dataclass(frozen=True)
class BinOp:
    op: str
    lhs: "Argument"
    rhs: "Argument"

    def __str__(self) -> str:
        return f"{self.lhs} {self.op} {self.rhs}"


@dataclass(frozen=True)
class JumpTarget:
    target: str

    def __str__(self) -> str:
        return f".{self.target}"


Argument = Union[
    Register, AsmGlobalSymbol, AsmAddressMode, Macro, AsmLiteral, BinOp, JumpTarget
]


@dataclass(frozen=True)
class StackLocation:
    """
    Represents a word on the stack. Currently only used for pattern matching.
    `symbolic_offset` represents a label offset that is only used in patterns,
    to represent the "N" in arguments such as `(N+4)($sp)`.
    """

    offset: int
    symbolic_offset: Optional[str]

    def __str__(self) -> str:
        prefix = "" if self.symbolic_offset is None else f"{self.symbolic_offset}+"
        return f"{prefix}{self.offset}($sp)"

    def offset_as_arg(self) -> Argument:
        if self.symbolic_offset is None:
            return AsmLiteral(self.offset)
        if self.offset == 0:
            return AsmGlobalSymbol(self.symbolic_offset)
        return BinOp(
            lhs=AsmGlobalSymbol(self.symbolic_offset),
            op="+",
            rhs=AsmLiteral(self.offset),
        )

    @staticmethod
    def from_offset(offset: Argument) -> Optional["StackLocation"]:
        def align(x: int) -> int:
            return x & ~3

        if isinstance(offset, AsmLiteral):
            return StackLocation(
                offset=align(offset.value),
                symbolic_offset=None,
            )
        if isinstance(offset, AsmGlobalSymbol):
            return StackLocation(
                offset=0,
                symbolic_offset=offset.symbol_name,
            )
        if (
            isinstance(offset, BinOp)
            and offset.op in ("+", "-")
            and isinstance(offset.lhs, AsmGlobalSymbol)
            and isinstance(offset.rhs, AsmLiteral)
        ):
            base = offset.rhs.value
            if offset.op == "-":
                base = -base
            return StackLocation(
                offset=align(base),
                symbolic_offset=offset.lhs.symbol_name,
            )
        return None


Location = Union[Register, StackLocation]


@dataclass(frozen=True)
class AsmInstruction:
    mnemonic: str
    args: List[Argument]

    def __str__(self) -> str:
        if not self.args:
            return self.mnemonic
        args = ", ".join(str(arg) for arg in self.args)
        return f"{self.mnemonic} {args}"


@dataclass(frozen=True)
class InstructionMeta:
    # True if the original asm line was marked with a goto pattern
    emit_goto: bool
    # Asm source filename & line number
    filename: str
    lineno: int
    # True if the Instruction is not directly from the source asm
    synthetic: bool

    @staticmethod
    def missing() -> "InstructionMeta":
        return InstructionMeta(
            emit_goto=False, filename="<unknown>", lineno=0, synthetic=True
        )

    def derived(self) -> "InstructionMeta":
        return replace(self, synthetic=True)

    def loc_str(self) -> str:
        adj = "near" if self.synthetic else "at"
        return f"{adj} {self.filename} line {self.lineno}"


@dataclass(frozen=True)
class Instruction:
    mnemonic: str
    args: List[Argument]
    meta: InstructionMeta

    # Track register and stack dependencies
    # An Instruction evaluates by reading from `inputs`, invalidating `clobbers`,
    # then writing to `outputs` (in that order)
    inputs: List[Location]
    clobbers: List[Location]
    outputs: List[Location]

    jump_target: Optional[Union[JumpTarget, Register]] = None
    function_target: Optional[Union[AsmGlobalSymbol, Register]] = None
    is_conditional: bool = False
    is_return: bool = False

    # These are for MIPS. `is_branch_likely` refers to branch instructions which
    # execute their delay slot only if the branch *is* taken. (Maybe these two
    # bools should be merged into a 3-valued enum?)
    has_delay_slot: bool = False
    is_branch_likely: bool = False

    # True if the Instruction was part of a matched IR pattern, but not elided
    in_pattern: bool = False

    def is_jump(self) -> bool:
        return self.jump_target is not None or self.is_return

    def __str__(self) -> str:
        if not self.args:
            return self.mnemonic
        args = ", ".join(str(arg) for arg in self.args)
        return f"{self.mnemonic} {args}"

    def arch_mnemonic(self, arch: "ArchAsm") -> str:
        """Combine architecture name with mnemonic for pattern matching"""
        return f"{arch.arch}:{self.mnemonic}"


class ArchAsmParsing(abc.ABC):
    """Arch-specific information needed to parse asm."""

    all_regs: List[Register]
    aliased_regs: Dict[str, Register]

    @abc.abstractmethod
    def normalize_instruction(self, instr: AsmInstruction) -> AsmInstruction:
        ...


class ArchAsm(ArchAsmParsing):
    """Arch-specific information that relates to the asm level. Extends the above."""

    arch: Target.ArchEnum

    stack_pointer_reg: Register
    frame_pointer_reg: Optional[Register]
    return_address_reg: Register

    base_return_regs: List[Register]
    all_return_regs: List[Register]
    argument_regs: List[Register]
    simple_temp_regs: List[Register]
    temp_regs: List[Register]
    saved_regs: List[Register]
    all_regs: List[Register]

    @abc.abstractmethod
    def missing_return(self) -> List[Instruction]:
        ...

    @abc.abstractmethod
    def parse(
        self, mnemonic: str, args: List[Argument], meta: InstructionMeta
    ) -> Instruction:
        ...


class NaiveParsingArch(ArchAsmParsing):
    """A fake arch that can parse asm in a naive fashion. Used by the pattern matching
    machinery to reduce arch dependence."""

    all_regs: List[Register] = []
    aliased_regs: Dict[str, Register] = {}

    def normalize_instruction(self, instr: AsmInstruction) -> AsmInstruction:
        return instr


@dataclass
class RegFormatter:
    """Converts register names used in input assembly to the internal register representation,
    saves the input's names, and converts back to the input's names for the output."""

    used_names: Dict[Register, str] = field(default_factory=dict)

    def parse(self, reg_name: str, arch: ArchAsmParsing) -> Register:
        return arch.aliased_regs.get(reg_name, Register(reg_name))

    def parse_and_store(self, reg_name: str, arch: ArchAsmParsing) -> Register:
        internal_reg = arch.aliased_regs.get(reg_name, Register(reg_name))
        existing_reg_name = self.used_names.get(internal_reg)
        if existing_reg_name is None:
            self.used_names[internal_reg] = reg_name
        elif existing_reg_name != reg_name:
            raise DecompFailure(
                f"Source uses multiple names for {internal_reg} ({existing_reg_name}, {reg_name})"
            )
        return internal_reg

    def format(self, reg: Register) -> str:
        return self.used_names.get(reg, reg.register_name)


valid_word = string.ascii_letters + string.digits + "_$"
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


def parse_number(elems: List[str]) -> int:
    number_str = parse_word(elems, valid_number)
    if number_str[0] == "0":
        assert len(number_str) == 1 or number_str[1] in "xX"
    ret = int(number_str, 0)
    return ret


def constant_fold(arg: Argument) -> Argument:
    if not isinstance(arg, BinOp):
        return arg
    lhs = constant_fold(arg.lhs)
    rhs = constant_fold(arg.rhs)
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
    return arg


def replace_bare_reg(
    arg: Argument, arch: ArchAsmParsing, reg_formatter: RegFormatter
) -> Argument:
    """If `arg` is an AsmGlobalSymbol whose name matches a known or aliased register,
    convert it into a Register and return it. Otherwise, return the original `arg`."""
    if isinstance(arg, AsmGlobalSymbol):
        reg_name = arg.symbol_name
        if Register(reg_name) in arch.all_regs or reg_name in arch.aliased_regs:
            return reg_formatter.parse_and_store(reg_name, arch)
    return arg


def get_jump_target(label: Argument) -> JumpTarget:
    if isinstance(label, AsmGlobalSymbol):
        return JumpTarget(label.symbol_name)
    assert isinstance(label, JumpTarget), "invalid branch target"
    return label


# Main parser.
def parse_arg_elems(
    arg_elems: List[str], arch: ArchAsmParsing, reg_formatter: RegFormatter
) -> Optional[Argument]:
    value: Optional[Argument] = None

    def expect(n: str) -> str:
        assert arg_elems, f"Expected one of {list(n)}, but reached end of string"
        g = arg_elems.pop(0)
        assert g in n, f"Expected one of {list(n)}, got {g} (rest: {arg_elems})"
        return g

    while arg_elems:
        tok: str = arg_elems[0]
        if tok.isspace():
            # Ignore whitespace.
            arg_elems.pop(0)
        elif tok == ",":
            expect(",")
            break
        elif tok == "$":
            # Register.
            assert value is None
            word = parse_word(arg_elems)
            reg = word[1:]
            if "$" in reg:
                # If there is a second $ in the word, it's a symbol
                value = AsmGlobalSymbol(word)
            else:
                value = Register(reg)
                value = reg_formatter.parse_and_store(value.register_name, arch)
        elif tok == ".":
            # Either a jump target (i.e. a label), or a section reference.
            assert value is None
            arg_elems.pop(0)
            word = parse_word(arg_elems)
            if word in ["data", "rodata", "bss", "text"]:
                value = asm_section_global_symbol(word, 0)
            else:
                value = JumpTarget(word)
        elif tok == "%":
            # A macro (i.e. %hi(...) or %lo(...)).
            assert value is None
            arg_elems.pop(0)
            macro_name = parse_word(arg_elems)
            assert macro_name in ("hi", "lo")
            expect("(")
            # Get the argument of the macro (which must exist).
            m = parse_arg_elems(arg_elems, arch, reg_formatter)
            assert m is not None
            m = constant_fold(m)
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
            expect("(")
            # Get what is being dereferenced.
            rhs = parse_arg_elems(arg_elems, arch, reg_formatter)
            assert rhs is not None
            expect(")")
            if isinstance(rhs, BinOp):
                # Binary operation.
                assert value is None
                value = constant_fold(rhs)
            else:
                # Address mode.
                rhs = replace_bare_reg(rhs, arch, reg_formatter)
                if rhs == AsmLiteral(0):
                    rhs = Register("zero")
                assert isinstance(rhs, Register)
                value = AsmAddressMode(value or AsmLiteral(0), rhs)
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
            value = AsmGlobalSymbol(word)
        elif tok in "<>+-&*":
            # Binary operators, used e.g. to modify global symbols or constants.
            assert isinstance(value, (AsmLiteral, AsmGlobalSymbol, BinOp))

            if tok in "<>":
                # bitshifts
                expect(tok)
                expect(tok)
                op = tok + tok
            else:
                op = expect("&+-*")

            if tok == "-" and arg_elems[0] == "_":
                # Parse `sym-_SDA_BASE_` as a Macro, equivalently to `sym@sda21`
                reloc_name = parse_word(arg_elems)
                if reloc_name not in ("_SDA_BASE_", "_SDA2_BASE_"):
                    raise DecompFailure(
                        f"Unexpected symbol {reloc_name} in subtraction expression"
                    )
                value = Macro("sda21", value)
            else:
                rhs = parse_arg_elems(arg_elems, arch, reg_formatter)
                assert rhs is not None
                if isinstance(rhs, BinOp) and rhs.op == "*":
                    rhs = constant_fold(rhs)
                if isinstance(rhs, BinOp) and isinstance(
                    constant_fold(rhs), AsmLiteral
                ):
                    raise DecompFailure(
                        "Math is too complicated for mips_to_c. Try adding parentheses."
                    )
                if isinstance(rhs, AsmLiteral) and isinstance(
                    value, AsmSectionGlobalSymbol
                ):
                    value = asm_section_global_symbol(
                        value.section_name, value.addend + rhs.value
                    )
                else:
                    value = BinOp(op, value, rhs)
        elif tok == "@":
            # A relocation (e.g. (...)@ha or (...)@l).
            arg_elems.pop(0)
            reloc_name = parse_word(arg_elems)
            assert reloc_name in ("h", "ha", "l", "sda2", "sda21")
            assert value
            value = Macro(reloc_name, value)
        else:
            assert False, f"Unknown token {tok} in {arg_elems}"

    return value


def parse_args(
    args: str, arch: ArchAsmParsing, reg_formatter: RegFormatter
) -> List[Argument]:
    arg_elems: List[str] = list(args.strip())
    output = []
    while arg_elems:
        ret = parse_arg_elems(arg_elems, arch, reg_formatter)
        assert ret is not None
        output.append(replace_bare_reg(constant_fold(ret), arch, reg_formatter))
    return output


def parse_asm_instruction(
    line: str, arch: ArchAsmParsing, reg_formatter: RegFormatter
) -> AsmInstruction:
    # First token is instruction name, rest is args.
    line = line.strip()
    mnemonic, _, args_str = line.partition(" ")
    # Parse arguments.
    args = parse_args(args_str, arch, reg_formatter)
    instr = AsmInstruction(mnemonic, args)
    return arch.normalize_instruction(instr)


def parse_instruction(
    line: str, meta: InstructionMeta, arch: ArchAsm, reg_formatter: RegFormatter
) -> Instruction:
    try:
        base = parse_asm_instruction(line, arch, reg_formatter)
        return arch.parse(base.mnemonic, base.args, meta)
    except Exception:
        raise DecompFailure(f"Failed to parse instruction {meta.loc_str()}: {line}")


@dataclass
class InstrProcessingFailure(Exception):
    instr: Instruction

    def __str__(self) -> str:
        return f"Error while processing instruction:\n{self.instr}"


@contextmanager
def current_instr(instr: Instruction) -> Iterator[None]:
    """Mark an instruction as being the one currently processed, for the
    purposes of error messages. Use like |with current_instr(instr): ...|"""
    try:
        yield
    except Exception as e:
        raise InstrProcessingFailure(instr) from e
