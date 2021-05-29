import re
import struct
import typing
from pathlib import Path
from typing import Callable, Dict, List, Match, Optional, Set, Tuple, TypeVar, Union

import attr

from .error import DecompFailure
from .options import Options
from .parse_instruction import Instruction, InstructionMeta, parse_instruction


@attr.s(frozen=True)
class Label:
    name: str = attr.ib()

    def __str__(self) -> str:
        return f"  .{self.name}:"


@attr.s
class Function:
    name: str = attr.ib()
    body: List[Union[Instruction, Label]] = attr.ib(factory=list)

    def new_label(self, name: str) -> None:
        label = Label(name)
        if self.body and self.body[-1] == label:
            # Skip repeated labels
            return
        self.body.append(label)

    def new_instruction(self, instruction: Instruction) -> None:
        self.body.append(instruction)

    def bodyless_copy(self) -> "Function":
        return Function(name=self.name)

    def __str__(self) -> str:
        body = "\n".join(str(item) for item in self.body)
        return f"glabel {self.name}\n{body}"


@attr.s
class RodataEntry:
    data: List[Union[str, bytes]] = attr.ib(factory=list)
    is_string: bool = attr.ib(default=False)


@attr.s
class Rodata:
    values: Dict[str, RodataEntry] = attr.ib(factory=dict)
    mentioned_labels: Set[str] = attr.ib(factory=set)

    def merge_into(self, other: "Rodata") -> None:
        for (sym, value) in self.values.items():
            other.values[sym] = value
        for label in self.mentioned_labels:
            other.mentioned_labels.add(label)


@attr.s
class MIPSFile:
    filename: str = attr.ib()
    functions: List[Function] = attr.ib(factory=list)
    rodata: Rodata = attr.ib(factory=Rodata)
    current_function: Optional[Function] = attr.ib(default=None, repr=False)
    current_rodata: RodataEntry = attr.ib(factory=RodataEntry)

    def new_function(self, name: str) -> None:
        self.current_function = Function(name=name)
        self.functions.append(self.current_function)

    def new_instruction(self, instruction: Instruction) -> None:
        if self.current_function is None:
            # Allow (and ignore) nop instructions in the .text
            # section before any function labels
            if instruction.mnemonic == "nop":
                return
            else:
                raise DecompFailure(
                    "unsupported non-nop instruction outside of function"
                )
        self.current_function.new_instruction(instruction)

    def new_label(self, label_name: str) -> None:
        assert self.current_function is not None
        self.current_function.new_label(label_name)

    def new_rodata_label(self, symbol_name: str) -> None:
        self.current_rodata = RodataEntry()
        self.rodata.values[symbol_name] = self.current_rodata

    def new_rodata_sym(self, sym: str) -> None:
        self.current_rodata.data.append(sym)
        self.rodata.mentioned_labels.add(sym.lstrip("."))

    def new_rodata_bytes(self, data: bytes, *, is_string: bool = False) -> None:
        if not self.current_rodata.data and is_string:
            self.current_rodata.is_string = True
        if self.current_rodata.data and isinstance(self.current_rodata.data[-1], bytes):
            self.current_rodata.data[-1] += data
        else:
            self.current_rodata.data.append(data)

    def __str__(self) -> str:
        functions_str = "\n\n".join(str(function) for function in self.functions)
        return f"# {self.filename}\n{functions_str}"


def parse_ascii_directive(line: str, z: bool) -> bytes:
    # This is wrong wrt encodings; the assembler really operates on bytes and
    # not chars. But for our purposes it should be good enough.
    in_quote = False
    num_parts = 0
    ret: List[bytes] = []
    i = 0
    digits = "0123456789"
    while i < len(line):
        c = line[i]
        i += 1
        if not in_quote:
            if c == '"':
                in_quote = True
                num_parts += 1
        else:
            if c == '"':
                in_quote = False
                if z:
                    ret.append(b"\0")
                continue
            if c != "\\":
                ret.append(c.encode("utf-8"))
                continue
            if i == len(line):
                raise DecompFailure(
                    "backslash at end of .ascii line not supported: " + line
                )
            c = line[i]
            i += 1
            char_escapes = {
                "b": b"\b",
                "f": b"\f",
                "n": b"\n",
                "r": b"\r",
                "t": b"\t",
                "v": b"\v",
            }
            if c in char_escapes:
                ret.append(char_escapes[c])
            elif c == "x":
                # hex literal, consume any number of hex chars, possibly none
                value = 0
                while i < len(line) and line[i] in digits + "abcdefABCDEF":
                    value = value * 16 + int(line[i], 16)
                    i += 1
                ret.append(bytes([value & 0xFF]))
            elif c in digits:
                # Octal literal, consume up to two more digits.
                # Using just the digits 0-7 would be more sane, but this matches GNU as.
                it = 0
                value = int(c)
                while i < len(line) and line[i] in digits and it < 2:
                    value = value * 8 + int(line[i])
                    i += 1
                    it += 1
                ret.append(bytes([value & 0xFF]))
            else:
                ret.append(c.encode("utf-8"))

    if in_quote:
        raise DecompFailure("unterminated string literal: " + line)
    if num_parts == 0:
        raise DecompFailure(".ascii with no string: " + line)
    return b"".join(ret)


def parse_file(f: typing.TextIO, options: Options) -> MIPSFile:
    filename = Path(f.name).name
    mips_file: MIPSFile = MIPSFile(filename)
    defines: Dict[str, int] = options.preproc_defines
    ifdef_level: int = 0
    ifdef_levels: List[int] = []
    curr_section = ".text"

    # https://stackoverflow.com/a/241506
    def re_comment_replacer(match: Match[str]) -> str:
        s = match.group(0)
        if s[0] in "/# \t":
            return " "
        else:
            return s

    re_comment_or_string = re.compile(r'#.*|/\*.*?\*/|"(?:\\.|[^\\"])*"')
    re_whitespace_or_string = re.compile(r'\s+|"(?:\\.|[^\\"])*"')
    re_local_glabel = re.compile("L(_U_)?[0-9A-F]{8}")
    re_local_label = re.compile("loc_|locret_|def_")
    re_label = re.compile(r"([a-zA-Z0-9_.]+):")

    T = TypeVar("T")

    def try_parse(parser: Callable[[], T], directive: str) -> T:
        try:
            return parser()
        except ValueError:
            raise DecompFailure(f"Could not parse rodata {directive}: {line}")

    for lineno, line in enumerate(f, 1):
        # Check for goto markers before stripping comments
        emit_goto = any(pattern in line for pattern in options.goto_patterns)

        # Strip comments and whitespace (but not within strings)
        line = re.sub(re_comment_or_string, re_comment_replacer, line)
        line = re.sub(re_whitespace_or_string, re_comment_replacer, line)
        line = line.strip()

        def process_label(label: str, *, glabel: bool) -> None:
            if curr_section == ".rodata":
                mips_file.new_rodata_label(label)
            elif curr_section == ".text":
                re_local = re_local_glabel if glabel else re_local_label
                if label.startswith("."):
                    if mips_file.current_function is None:
                        raise DecompFailure(f"Label {label} is not within a function!")
                    mips_file.new_label(label.lstrip("."))
                elif re_local.match(label) and mips_file.current_function is not None:
                    # Don't treat labels as new functions if they follow a
                    # specific naming pattern. This is used for jump table
                    # targets in both IDA and old n64split output.
                    # We skip this behavior for the very first label in the
                    # file though, to avoid crashes due to unidentified
                    # functions. (Should possibly be generalized to cover any
                    # glabel that has a branch that goes across?)
                    mips_file.new_label(label)
                else:
                    mips_file.new_function(label)

        # Check for labels
        while True:
            g = re_label.match(line)
            if not g:
                break

            label = g.group(1)
            if ifdef_level == 0:
                process_label(label, glabel=False)

            line = line[len(label) + 1 :].strip()

        if not line:
            continue

        if line.startswith("."):
            # Assembler directive.
            if line.startswith(".ifdef") or line.startswith(".ifndef"):
                macro_name = line.split()[1]
                if macro_name not in defines:
                    defines[macro_name] = 0
                    print(
                        f"Note: assuming {macro_name} is unset for .ifdef, "
                        f"pass -D{macro_name}/-U{macro_name} to set/unset explicitly."
                    )
                level = defines[macro_name]
                if line.startswith(".ifdef"):
                    level = 1 - level
                ifdef_level += level
                ifdef_levels.append(level)
            elif line.startswith(".else"):
                level = ifdef_levels.pop()
                ifdef_level -= level
                level = 1 - level
                ifdef_level += level
                ifdef_levels.append(level)
            elif line.startswith(".endif"):
                ifdef_level -= ifdef_levels.pop()
            elif line.startswith(".macro"):
                ifdef_level += 1
            elif line.startswith(".endm"):
                ifdef_level -= 1
            elif ifdef_level == 0:
                if line.startswith(".section"):
                    curr_section = line.split(" ")[1].split(",")[0]
                    if curr_section == ".late_rodata":
                        curr_section = ".rodata"
                elif (
                    line.startswith(".rdata")
                    or line.startswith(".rodata")
                    or line.startswith(".late_rodata")
                ):
                    curr_section = ".rodata"
                elif line.startswith(".text"):
                    curr_section = ".text"
                elif curr_section == ".rodata":
                    if line.startswith(".word"):
                        for w in line[5:].split(","):
                            w = w.strip()
                            if not w or w[0].isdigit():
                                ival = try_parse(lambda: int(w, 0), ".word")
                                mips_file.new_rodata_bytes(struct.pack(">I", ival))
                            else:
                                mips_file.new_rodata_sym(w)
                    elif line.startswith(".byte"):
                        for w in line[5:].split(","):
                            ival = try_parse(lambda: int(w.strip(), 0), ".byte")
                            mips_file.new_rodata_bytes(bytes([ival]))
                    elif line.startswith(".float"):
                        for w in line[6:].split(","):
                            fval = try_parse(lambda: float(w.strip()), ".float")
                            mips_file.new_rodata_bytes(struct.pack(">f", fval))
                    elif line.startswith(".double"):
                        for w in line[7:].split(","):
                            fval = try_parse(lambda: float(w.strip()), ".double")
                            mips_file.new_rodata_bytes(struct.pack(">d", fval))
                    elif line.startswith(".asci"):
                        z = line.startswith(".asciz") or line.startswith(".asciiz")
                        mips_file.new_rodata_bytes(
                            parse_ascii_directive(line, z), is_string=True
                        )
        elif ifdef_level == 0:
            if line.startswith("glabel"):
                process_label(line.split()[1], glabel=True)

            elif curr_section == ".text":
                meta = InstructionMeta(
                    emit_goto=emit_goto,
                    filename=filename,
                    lineno=lineno,
                    synthetic=False,
                )
                instr: Instruction = parse_instruction(line, meta)
                mips_file.new_instruction(instr)

    return mips_file
