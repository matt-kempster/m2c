from __future__ import annotations
import csv
from dataclasses import dataclass, field
from enum import Enum
import re
import struct
import typing
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Match,
    NoReturn,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .error import DecompFailure
from .options import CodingStyle, Options
from .asm_instruction import (
    Argument,
    AsmGlobalSymbol,
    AsmLiteral,
    AsmState,
    RegFormatter,
    parse_arg,
    traverse_arg,
)
from .instruction import (
    ArchAsm,
    Instruction,
    InstructionMeta,
    parse_instruction,
)


RE_COMMA_OR_STRING = re.compile(r',|"(?:\\.|[^\\"])*"')


@dataclass(frozen=True)
class Label:
    # Various pattern matching code assumes that there cannot be consecutive
    # labels, and to deal with this we allow for consecutive labels to be
    # merged together. As a consequence, we allow a Label to have more than one
    # name. When we need a single name to refer to one, we use the first one.
    names: List[str]

    def __str__(self) -> str:
        return self.names[0]


@dataclass
class Function:
    name: str
    body: List[Union[Instruction, Label]] = field(default_factory=list)
    reg_formatter: RegFormatter = field(default_factory=RegFormatter)

    def new_label(self, name: str) -> None:
        label = Label([name])
        if self.body and self.body[-1] == label:
            # Skip repeated labels
            return
        self.body.append(label)

    def new_instruction(self, instruction: Instruction) -> None:
        self.body.append(instruction)

    def bodyless_copy(self) -> Function:
        return Function(
            name=self.name,
            reg_formatter=self.reg_formatter,
        )

    def __str__(self) -> str:
        body = "\n".join(
            str(item) if isinstance(item, Instruction) else f"  {item}:"
            for item in self.body
        )
        return f"glabel {self.name}\n{body}"


@dataclass
class AsmSymbolicData:
    data: Argument
    size: int

    def as_symbol_without_addend(self) -> Optional[str]:
        if isinstance(self.data, AsmGlobalSymbol) and self.size == 4:
            return self.data.symbol_name
        return None


@dataclass
class AsmDataEntry:
    sort_order: Tuple[str, int]
    data: List[Union[bytes, AsmSymbolicData]] = field(default_factory=list)
    is_string: bool = False
    is_readonly: bool = False
    is_bss: bool = False
    is_text: bool = False
    is_jtbl: bool = False

    # Mutable state:
    used_as_literal: bool = False

    def size_range_bytes(self) -> Tuple[int, int]:
        """Return the range of possible sizes, if padding were stripped."""
        # TODO: The data address could be used to only strip padding
        # that ends on 16-byte boundaries and is at the end of a section
        max_size = 0
        for x in self.data:
            if isinstance(x, AsmSymbolicData):
                max_size += x.size
            else:
                max_size += len(x)

        padding_size = 0
        if self.data and isinstance(self.data[-1], bytes):
            assert len(self.data) == 1 or isinstance(self.data[-2], AsmSymbolicData)
            for b in self.data[-1][::-1]:
                if b != 0:
                    break
                padding_size += 1
        padding_size = min(padding_size, 15)
        assert padding_size <= max_size

        # Assume the size is at least 1 byte, unless `max_size == 0`
        if max_size == padding_size and max_size != 0:
            return 1, max_size
        return max_size - padding_size, max_size

    def data_at_offset(
        self, offset: int, size: int
    ) -> Optional[Union[bytes, AsmSymbolicData]]:
        for data in self.data:
            subsize = len(data) if isinstance(data, bytes) else data.size
            if offset >= subsize:
                offset -= subsize
                continue
            if isinstance(data, bytes):
                data = data[offset : offset + size]
                if len(data) != size:
                    return None
                return data
            else:
                if data.size != size:
                    return None
                return data
        return None


@dataclass
class AsmData:
    values: Dict[str, AsmDataEntry] = field(default_factory=dict)
    mentioned_labels: Set[str] = field(default_factory=set)

    def merge_into(self, other: AsmData) -> None:
        for sym, value in self.values.items():
            other.values[sym] = value
        for label in self.mentioned_labels:
            other.mentioned_labels.add(label)

    def is_likely_char(self, c: int) -> bool:
        return 0x20 <= c < 0x7F or c in (0, 7, 8, 9, 10, 13, 27)

    def detect_heuristic_strings(self) -> None:
        for ent in self.values.values():
            if (
                ent.is_readonly
                and len(ent.data) == 1
                and isinstance(ent.data[0], bytes)
                and len(ent.data[0]) > 1
                and ent.data[0][0] != 0
                and all(self.is_likely_char(x) for x in ent.data[0])
            ):
                ent.is_string = True


@dataclass
class AsmFile:
    filename: str
    reg_formatter: RegFormatter
    functions: List[Function] = field(default_factory=list)
    asm_data: AsmData = field(default_factory=AsmData)
    current_function: Optional[Function] = field(default=None, repr=False)
    current_data: Optional[AsmDataEntry] = field(default=None)

    def new_function(self, name: str) -> None:
        self.current_function = Function(name=name, reg_formatter=self.reg_formatter)
        self.functions.append(self.current_function)

    def new_instruction(self, instruction: Instruction) -> None:
        if self.current_function is None:
            # Allow (and ignore) nop instructions in the .text
            # section before any function labels
            if instruction.mnemonic == "nop":
                return
            else:
                raise DecompFailure(
                    f"unsupported non-nop instruction outside of function ({instruction})"
                )
        self.current_function.new_instruction(instruction)
        self.current_data = None

    def new_label(self, label_name: str) -> None:
        assert self.current_function is not None
        self.current_function.new_label(label_name)

    def new_data_label(
        self,
        symbol_name: str,
        is_readonly: bool = False,
        is_bss: bool = False,
        is_text: bool = False,
    ) -> None:
        sort_order = (self.filename, len(self.asm_data.values))
        self.current_data = AsmDataEntry(
            sort_order, is_readonly=is_readonly, is_bss=is_bss, is_text=is_text
        )
        self.asm_data.values[symbol_name] = self.current_data

    def new_symbolic_data(self, data: Argument, size: int) -> None:
        if self.current_data is not None:
            self.current_data.data.append(AsmSymbolicData(data, size))
        for subexpr in traverse_arg(data):
            if isinstance(subexpr, AsmGlobalSymbol):
                self.asm_data.mentioned_labels.add(subexpr.symbol_name)

    def new_data_bytes(self, data: bytes, *, is_string: bool = False) -> None:
        if self.current_data is None:
            return
        if not self.current_data.data and is_string:
            self.current_data.is_string = True
        if self.current_data.data and isinstance(self.current_data.data[-1], bytes):
            self.current_data.data[-1] += data
        else:
            self.current_data.data.append(data)

    def __str__(self) -> str:
        functions_str = "\n\n".join(str(function) for function in self.functions)
        return f"# {self.filename}\n{functions_str}"


def split_quotable_arg_list(args: str) -> List[str]:
    """Split a string of comma-separated arguments, handling quotes"""
    reader = csv.reader(
        [args],
        delimiter=",",
        doublequote=False,
        escapechar="\\",
        quotechar='"',
        skipinitialspace=True,
    )
    return [a.strip() for a in next(reader)]


def split_arg_list(args: str) -> List[str]:
    if '"' not in args:
        return [a.strip() for a in args.split(",")]
    commas = [
        m.span()[0] for m in RE_COMMA_OR_STRING.finditer(args) if m.group(0) == ","
    ]
    return [
        args[a + 1 : b].strip() for a, b in zip([-1] + commas, commas + [len(args)])
    ]


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


def add_warning(warnings: List[str], new: str) -> None:
    if new not in warnings:
        warnings.append(new)


def parse_incbin(
    args: List[str], options: Options, warnings: List[str]
) -> Optional[bytes]:
    try:
        if len(args) == 3:
            filename = args[0]
            offset = int(args[1], 0)
            size = int(args[2], 0)
        elif len(args) == 1:
            filename = args[0]
            offset = 0
            size = -1
        else:
            raise ValueError
    except ValueError:
        raise DecompFailure(f"Could not parse asm_data .incbin directive: {args}")

    if not options.incbin_dirs:
        add_warning(
            warnings,
            f"Skipping .incbin directive for {filename}, pass --incbin-dir to set a search directory",
        )
        return None

    for incbin_dir in options.incbin_dirs:
        full_path = incbin_dir / filename
        try:
            with full_path.open("rb") as f:
                f.seek(offset)
                data = f.read(size)
        except OSError:
            continue
        except MemoryError:
            data = b""

        if size >= 0 and len(data) != size:
            add_warning(
                warnings,
                f"Unable to read {size} bytes from {full_path} at {offset:#x} (got {len(data)} bytes)",
            )
            return None
        return data

    add_warning(
        warnings,
        f"Unable to find {filename} in any of {len(options.incbin_dirs)} search paths",
    )
    return None


def check_ifdef(
    macro_name: str,
    asm_state: AsmState,
    warnings: List[str],
    directive: str,
) -> bool:
    if macro_name not in asm_state.defines:
        asm_state.defines[macro_name] = None
        add_warning(
            warnings,
            f"Note: assuming {macro_name} is unset for {directive}, "
            f"pass -D{macro_name}/-U{macro_name} to set/unset explicitly.",
        )
    return asm_state.defines[macro_name] is not None


def eval_cpp_if(
    expr: str, asm_state: AsmState, warnings: List[str], directive: str
) -> bool:
    token_re = re.compile(r"[a-zA-Z0-9_]+|&&|\|\||.")
    tokens = [m.group() for m in token_re.finditer(expr) if not m.group().isspace()]

    i = 0

    def parse1() -> bool:
        nonlocal i
        tok = tokens[i]
        i += 1
        if tok == "!":
            return not parse1()
        if tok == "(":
            r = parse3()
            assert tokens[i] == ")"
            i += 1
            return r
        if tok == "defined":
            assert tokens[i] == "("
            w = tokens[i + 1]
            assert tokens[i + 2] == ")"
            i += 3
            return check_ifdef(w, asm_state, warnings, directive)
        if tok[0].isnumeric():
            return bool(int(tok, 0))
        else:
            value = asm_state.defines.get(tok, None)
            if value is None:
                check_ifdef(tok, asm_state, warnings, directive)
                return False
            return bool(value)

    def parse2() -> bool:
        nonlocal i
        r = parse1()
        while i < len(tokens) and tokens[i] == "&&":
            i += 1
            r &= parse1()
        return r

    def parse3() -> bool:
        nonlocal i
        r = parse2()
        while i < len(tokens) and tokens[i] == "||":
            i += 1
            r |= parse2()
        return r

    try:
        ret = parse3()
        assert i == len(tokens)
    except (AssertionError, IndexError, ValueError):
        raise DecompFailure(f"Failed to parse #if condition: {expr}")
    return ret


def parse_file(f: typing.TextIO, arch: ArchAsm, options: Options) -> AsmFile:
    filename = Path(f.name).name
    reg_formatter = RegFormatter()
    asm_file: AsmFile = AsmFile(filename, reg_formatter)
    curr_section = ".text"
    warnings: List[str] = []
    asm_state = AsmState(
        defines={
            # NULL is a non-standard but common asm macro that expands to 0
            "NULL": 0,
            **options.preproc_defines,
        },
        reg_formatter=reg_formatter,
    )

    # Each nested ifdef has an associated level which is 0 (active), 1 (inactive)
    # or 2 (inactive, but was active previously). Lines are only processed if the
    # sum of levels (`ifdef_level`) is 0.
    ifdef_level: int = 0
    ifdef_levels: List[int] = []

    # https://stackoverflow.com/a/241506
    def re_comment_replacer(match: Match[str]) -> str:
        s = match.group(0)
        if s[0] in "/#;@ \t":
            return " "
        else:
            return s

    re_whitespace_or_string = re.compile(r'\s+|"(?:\\.|[^\\"])*"')
    re_local_glabel = re.compile("L(_.*_)?[0-9A-F]{7,8}")
    re_local_label = re.compile(
        "loc_|locret_|def_|lbl_|LAB_|switchD_|jump_|_[0-9A-Fa-f]{7,8}(?:_.*)?$"
    )
    re_label = re.compile(r'(?:([a-zA-Z0-9_.$]+)|"([a-zA-Z0-9_.$<>@,-]+)"):')

    T = TypeVar("T")

    class LabelKind(Enum):
        GLOBAL = "global"
        LOCAL = "local"
        JUMP_TARGET = "jump_target"

    def fail_parse() -> NoReturn:
        raise DecompFailure(
            f"Could not parse asm_data {directive} in {curr_section}: {line}"
        )

    def try_parse(parser: Callable[[], T]) -> T:
        try:
            return parser()
        except Exception:
            fail_parse()

    def parse_int(w: str) -> int:
        value = try_parse(lambda: parse_arg(w, arch, asm_state))
        if not isinstance(value, AsmLiteral):
            fail_parse()
        return value.value

    def pack(fmt: str, val: Union[int, float]) -> bytes:
        endian = ">" if options.target.is_big_endian() else "<"
        return struct.pack(endian + fmt, val)

    def emit_word(w: str, size: int) -> None:
        value = try_parse(lambda: parse_arg(w, arch, asm_state))
        if isinstance(value, AsmLiteral):
            ival = value.value & ((1 << (size * 8)) - 1)
            if options.target.is_big_endian():
                data = ival.to_bytes(size, "big")
            else:
                data = ival.to_bytes(size, "little")
            asm_file.new_data_bytes(data)
        else:
            asm_file.new_symbolic_data(value, size)

    re_comment_or_string = re.compile(arch.re_comment + r'|/\*.*?\*/|"(?:\\.|[^\\"])*"')
    for lineno, line in enumerate(f, 1):
        # Check for goto markers before stripping comments
        emit_goto = any(pattern in line for pattern in options.goto_patterns)

        # Convert C preprocessor directives into asm ones
        is_c_preproc_command = False
        if line.split() and line.split()[0] in (
            "#ifdef",
            "#ifndef",
            "#else",
            "#if",
            "#elif",
            "#elifdef",
            "#elifndef",
            "#endif",
            "#define",
        ):
            is_c_preproc_command = True
            line = "." + line[1:]

        # Strip comments and whitespace (but not within strings)
        line = re.sub(re_comment_or_string, re_comment_replacer, line)
        line = re.sub(re_whitespace_or_string, re_comment_replacer, line)
        line = line.strip()

        def process_label(label: str, *, kind: LabelKind) -> None:
            if curr_section == ".rodata":
                asm_file.new_data_label(label, is_readonly=True)
            elif curr_section == ".data":
                asm_file.new_data_label(label)
            elif curr_section == ".bss":
                asm_file.new_data_label(label, is_bss=True)
            elif curr_section == ".text":
                asm_file.new_data_label(label, is_readonly=True, is_text=True)
                if label.startswith(".") or kind == LabelKind.JUMP_TARGET:
                    if asm_file.current_function is not None:
                        asm_file.new_label(label)
                elif (
                    re_local_glabel.match(label)
                    or (kind != LabelKind.GLOBAL and re_local_label.match(label))
                ) and asm_file.current_function is not None:
                    # Don't treat labels as new functions if they follow a
                    # specific naming pattern. This is used for jump table
                    # targets in both IDA and old n64split output.
                    # We skip this behavior for the very first label in the
                    # file though, to avoid crashes due to unidentified
                    # functions. (Should possibly be generalized to cover any
                    # glabel that has a branch that goes across?)
                    asm_file.new_label(label)
                else:
                    asm_file.new_function(label)

        # Check for labels
        while True:
            g = re_label.match(line)
            if not g:
                break

            label = g.group(1) or g.group(2)
            if ifdef_level == 0:
                process_label(label, kind=LabelKind.LOCAL)

            line = line[len(g.group(0)) :].strip()

        if not line:
            continue

        if "=" in line and not is_c_preproc_command:
            key, value = line.split("=", 1)
            key = key.strip()
            if " " not in key:
                line = f".set {key}, {value}"

        directive = line.split()[0]
        if directive.startswith("."):
            # Assembler directive.
            _, _, args_str = line.partition(" ")
            if is_c_preproc_command:
                line = "#" + line[1:]
                directive = line.split()[0]
            if directive in (".ifdef", ".ifndef", "#ifdef", "#ifndef"):
                active = check_ifdef(args_str, asm_state, warnings, directive)
                if directive[1:] == "ifndef":
                    active = not active
                level = 1 - int(active)
                ifdef_level += level
                ifdef_levels.append(level)
            elif directive == ".if":
                macro_name = line.split()[1]
                if macro_name == "0":
                    active = False
                elif macro_name == "1":
                    active = True
                else:
                    active = True
                    add_warning(warnings, f"Note: ignoring .if {macro_name} directive")
                level = 1 - int(active)
                ifdef_level += level
                ifdef_levels.append(level)
            elif directive == "#if":
                active = eval_cpp_if(args_str, asm_state, warnings, directive)
                level = 1 - int(active)
                ifdef_level += level
                ifdef_levels.append(level)
            elif directive in ("#elif", "#elifdef", "#elifndef", "#else", ".else"):
                level = ifdef_levels.pop()
                ifdef_level -= level
                if level == 1:
                    if directive == "#elifdef":
                        active = check_ifdef(args_str, asm_state, warnings, directive)
                    elif directive == "#elifndef":
                        active = not check_ifdef(
                            args_str, asm_state, warnings, directive
                        )
                    elif directive == "#elif":
                        active = eval_cpp_if(args_str, asm_state, warnings, directive)
                    else:
                        active = True
                    level = 1 - int(active)
                else:
                    level = 2
                ifdef_level += level
                ifdef_levels.append(level)
            elif directive in (".endif", "#endif"):
                ifdef_level -= ifdef_levels.pop()
            elif directive == ".macro":
                ifdef_level += 1
            elif directive == ".endm":
                ifdef_level -= 1
            elif directive == ".fn":
                args = split_quotable_arg_list(args_str)
                asm_file.new_function(args[0])
            elif ifdef_level == 0:
                if directive == ".section":
                    curr_section = line.split()[1].split(",")[0]
                    if curr_section in (".rdata", ".late_rodata", ".sdata2"):
                        curr_section = ".rodata"
                    elif curr_section == ".sdata":
                        curr_section = ".data"
                    elif curr_section.startswith(".text"):
                        curr_section = ".text"
                elif (
                    directive == ".rdata"
                    or directive == ".rodata"
                    or directive == ".late_rodata"
                ):
                    curr_section = ".rodata"
                elif directive == ".data":
                    curr_section = ".data"
                elif directive == ".bss":
                    curr_section = ".bss"
                elif directive == ".text":
                    curr_section = ".text"
                elif directive == ".set":
                    args = split_quotable_arg_list(args_str)
                    if len(args) == 1:
                        # ".set noreorder" or similar, just ignore
                        pass
                    elif len(args) == 2:
                        asm_state.defines[args[0]] = parse_int(args[1])
                    else:
                        raise DecompFailure(f"Could not parse {directive}: {line}")
                elif directive == "#define":
                    args = args_str.split(None, 1)
                    if len(args) == 2:
                        asm_state.defines[args[0]] = parse_int(args[1])
                    else:
                        asm_state.defines[args[0]] = 1
                elif directive == ".syntax":
                    if args_str.strip() == "unified":
                        asm_state.is_unified = True
                    elif args_str.strip() == "divided":
                        asm_state.is_unified = False
                elif directive in (".thumb", ".thumb_func"):
                    asm_state.is_thumb = True
                elif directive == ".arm":
                    asm_state.is_thumb = False
                elif directive == ".code":
                    if args_str.strip() == "16":
                        asm_state.is_thumb = True
                    elif args_str.strip() == "32":
                        asm_state.is_thumb = False
                elif curr_section in (".rodata", ".data", ".bss", ".text"):
                    if directive in (".word", ".gpword", ".4byte"):
                        args = split_arg_list(args_str)
                        for w in args:
                            emit_word(w, 4)
                    elif directive == ".rel":
                        # .rel is a common dtk disassembler macro used with jump tables.
                        # ".rel name, label" expands to ".4byte name + (label - name)"
                        args = split_arg_list(args_str)
                        assert len(args) == 2
                        emit_word(args[1], 4)
                    elif directive == ".obj":
                        # dtk disassembler label format
                        args = split_quotable_arg_list(args_str)
                        assert len(args) == 2
                        kind = (
                            LabelKind.LOCAL if args[1] == "local" else LabelKind.GLOBAL
                        )
                        process_label(args[0], kind=kind)
                    elif directive in (".short", ".half", ".2byte"):
                        args = split_arg_list(args_str)
                        for w in args:
                            emit_word(w, 2)
                    elif directive == ".byte":
                        args = split_arg_list(args_str)
                        for w in args:
                            emit_word(w, 1)
                    elif directive == ".float":
                        args = split_arg_list(args_str)
                        for w in args:
                            fval = try_parse(lambda: float(w))
                            asm_file.new_data_bytes(pack("f", fval))
                    elif directive == ".double":
                        args = split_arg_list(args_str)
                        for w in args:
                            fval = try_parse(lambda: float(w))
                            asm_file.new_data_bytes(pack("d", fval))
                    elif directive in (
                        ".asci",
                        ".asciz",
                        ".ascii",
                        ".asciiz",
                        ".string",
                    ):
                        z = directive.endswith("z") or directive == ".string"
                        asm_file.new_data_bytes(
                            parse_ascii_directive(line, z), is_string=True
                        )
                    elif directive in (".space", ".skip"):
                        args = split_arg_list(args_str)
                        if len(args) == 2:
                            fill = parse_int(args[1]) & 0xFF
                        elif len(args) == 1:
                            fill = 0
                        else:
                            raise DecompFailure(
                                f"Could not parse asm_data {directive} in {curr_section}: {line}"
                            )
                        size = parse_int(args[0])
                        asm_file.new_data_bytes(bytes([fill] * size))
                    elif line.startswith(".incbin"):
                        args = split_quotable_arg_list(args_str)
                        data = parse_incbin(args, options, warnings)
                        if data is not None:
                            asm_file.new_data_bytes(data)

        elif ifdef_level == 0:
            if directive == "jlabel":
                _, _, args_str = line.partition(" ")
                args = split_quotable_arg_list(args_str)
                if args:
                    process_label(args[0], kind=LabelKind.JUMP_TARGET)

            elif directive in (
                "glabel",
                "dlabel",
                "arm_func_start",
                "thumb_func_start",
                "non_word_aligned_thumb_func_start",
                "ARM_FUNC_START",
                "THUMB_FUNC_START",
                "NON_WORD_ALIGNED_THUMB_FUNC_START",
            ):
                if "thumb" in directive.lower():
                    asm_state.is_thumb = True
                elif "arm" in directive.lower():
                    asm_state.is_thumb = False
                _, _, args_str = line.partition(" ")
                args = split_quotable_arg_list(args_str)
                if args:
                    process_label(args[0], kind=LabelKind.GLOBAL)

            elif directive in (
                "arm_func_end",
                "thumb_func_end",
                "ARM_FUNC_END",
                "THUMB_FUNC_END",
                "endlabel",
                "enddlabel",
                "alabel",
                "nonmatching",
            ):
                pass

            elif curr_section == ".text":
                meta = InstructionMeta(
                    emit_goto=emit_goto,
                    filename=filename,
                    lineno=lineno,
                    synthetic=False,
                )
                ins = parse_instruction(line, meta, arch, asm_state)
                asm_file.new_instruction(ins)

    if warnings and options.coding_style.comment_style != CodingStyle.CommentStyle.NONE:
        print("/*")
        print("\n".join(warnings))
        print("*/")

    return asm_file
