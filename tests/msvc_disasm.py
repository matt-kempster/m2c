from __future__ import annotations
import re
import struct
import sys
from dataclasses import dataclass
from typing import BinaryIO, Dict, List, Optional, Set, TextIO, Tuple, Union

import capstone as cs

from coff_file import (
    IMAGE_REL_I386_DIR32,
    IMAGE_REL_I386_DIR32NB,
    IMAGE_REL_I386_REL32,
    CoffFile,
    CoffRelocation,
    CoffSection,
    CoffSymbol,
)


@dataclass
class Insn:
    """A decoded machine instruction."""

    insn: cs.CsInsn
    address: int
    raw: bytes


@dataclass
class DataLong:
    """A 4-byte relocated data word living in `.text` (a jump-table entry)."""

    address: int
    reloc: CoffRelocation


@dataclass
class DataByte:
    """A single raw data byte living in `.text` (a 2-level switch byte map)."""

    address: int
    value: int


# One decoded `.text` element: either a real instruction or in-`.text` data
# (jump tables and case-mapping byte maps that MSVC emits after the code).
Element = Union[Insn, DataLong, DataByte]


RE_VALID_UNQUOTED = re.compile(r"^[A-Za-z0-9_.$]+$")

# MSVC emits float/double literals as read-only constants with a mangled name
# encoding the value: `__real@<hex>` (the raw 4- or 8-byte IEEE-754 value) or,
# as MSVC6 does, `__real@<size>@<80-bit-x87-extended-hex>` where <size> is 4 or
# 8. The name is not a valid C identifier and the raw bytes carry no type, so
# the disassembler decodes such symbols into a `.float`/`.double` data label
# under a clean, value-derived name that m2c renders as the real constant.
RE_REAL = re.compile(r"^__real@(?:([48])@)?([0-9a-fA-F]+)$")


def real_constant(name: str, data: bytes) -> Optional[Tuple[str, str, str, int]]:
    """If `name` is an MSVC `__real@` float/double constant and `data` (the
    section bytes starting at the symbol) round-trips through a decimal literal,
    return `(clean_name, directive, literal, size)`; otherwise None.

    The clean name is derived from the raw IEEE bits so identical constants map
    to one label and every reference agrees without needing the bytes."""
    match = RE_REAL.match(name)
    if match is None:
        return None
    size_field, hex_part = match.group(1), match.group(2)
    if size_field is not None:
        size = int(size_field)
    elif len(hex_part) == 8:
        size = 4
    elif len(hex_part) == 16:
        size = 8
    else:
        return None
    if len(data) < size:
        return None
    raw = data[:size]
    if size == 4:
        (bits,) = struct.unpack("<I", raw)
        (value,) = struct.unpack("<f", raw)
        clean_name = f"_real_{bits:08x}"
        directive = ".float"
    else:
        (bits,) = struct.unpack("<Q", raw)
        (value,) = struct.unpack("<d", raw)
        clean_name = f"_real_{bits:016x}"
        directive = ".double"
    literal = repr(value)
    try:
        packed = struct.pack("<f" if size == 4 else "<d", float(literal))
    except (ValueError, OverflowError):
        return None
    if packed != raw:
        # NaN/Inf or otherwise non-round-tripping: fall back to raw words.
        return None
    return clean_name, directive, literal, size


def rename_real_constants(
    sections: List[CoffSection],
) -> Dict[int, Dict[int, Tuple[str, str, int]]]:
    """Rename every `__real@` constant symbol in place to a clean identifier
    (CoffSymbol objects are shared with relocations, so references follow) and
    return, per section index, the offsets to emit as `.float`/`.double`."""
    per_section: Dict[int, Dict[int, Tuple[str, str, int]]] = {}
    for index, section in enumerate(sections):
        emit_map: Dict[int, Tuple[str, str, int]] = {}
        for offset, symbol in section.symbols.items():
            decoded = real_constant(symbol.name, section.data[offset:])
            if decoded is None:
                continue
            clean_name, directive, literal, size = decoded
            symbol.name = clean_name
            emit_map[offset] = (directive, literal, size)
        if emit_map:
            per_section[index] = emit_map
    return per_section


def address_label(addr: int) -> str:
    return f".L{addr:08X}"


def symbol_name(sym: CoffSymbol) -> str:
    return sym.name


def text_symbol_name(sym: CoffSymbol) -> str:
    if sym.section is not None and sym.section.name == ".text":
        decorated = re.fullmatch(r"[_@]([^@]+)@\d+", sym.name)
        if decorated is not None:
            return decorated.group(1)
        if sym.name.startswith("_") and not sym.name.startswith("__"):
            return sym.name[1:]
    return sym.name


def asm_name(name: str) -> str:
    if RE_VALID_UNQUOTED.fullmatch(name):
        return name
    escaped = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def relocation_target(
    reloc: CoffRelocation, labels: Set[int], *, for_relative: bool
) -> str:
    symbol = reloc.symbol
    offset = 0 if for_relative else reloc.symbol_offset
    if symbol.section is not None:
        target_offset = symbol.offset + reloc.symbol_offset
        if symbol.section.name == ".text":
            symbol_at_offset = symbol.section.symbols.get(target_offset)
            if symbol_at_offset is not None:
                return asm_name(text_symbol_name(symbol_at_offset))
            target_address = symbol.section.address + target_offset
            labels.add(target_address)
            return address_label(target_address)
        if target_offset != symbol.offset:
            offset = target_offset - symbol.offset

    base = asm_name(symbol_name(symbol))
    if offset == 0:
        return base
    sign = "+" if offset > 0 else "-"
    return f"{base} {sign} {abs(offset):#x}"


def split_operands(op_str: str) -> List[str]:
    operands: List[str] = []
    start = 0
    depth = 0
    for i, char in enumerate(op_str):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            operands.append(op_str[start:i].strip())
            start = i + 1
    if op_str:
        operands.append(op_str[start:].strip())
    return operands


# A displacement rendered by capstone at the end of a memory operand:
# either the whole expression (`[0x8]`) or a trailing `+`/`-` term
# (`[eax*4 + 0x8]`). Registers never end in a bare number on 32-bit x86,
# and scale factors (`eax*4`) are not preceded by a sign, so this cannot
# eat part of a register or scale term.
RE_TRAILING_DISPLACEMENT = re.compile(r"(?:^|\s*[+-]\s*)(?:0x[0-9a-fA-F]+|\d+)$")


def replace_memory_displacement(operand: str, label: str) -> str:
    """Substitute the relocation target for a memory operand's displacement.

    The displacement bytes are the relocation addend, and `relocation_target`
    already folds that addend into `label`, so the rendered displacement must
    be dropped entirely; keeping it would count the addend twice
    (e.g. `[4 + _ar + 0x4]` instead of `[_ar + 0x4]`).
    """
    start = operand.index("[")
    end = operand.rindex("]")
    inner = operand[start + 1 : end].strip()
    remainder = RE_TRAILING_DISPLACEMENT.sub("", inner).strip()
    if not remainder:
        new_inner = label
    else:
        new_inner = f"{remainder} + {label}"
    return operand[: start + 1] + new_inner + operand[end:]


def instruction_to_text(insn: cs.CsInsn, section: CoffSection, labels: Set[int]) -> str:
    offset = insn.address - section.address
    reloc_by_operand_offset: Dict[int, CoffRelocation] = {
        reloc_offset - offset: reloc
        for reloc_offset, reloc in section.relocations.items()
        if offset <= reloc_offset < offset + insn.size
    }

    operands = split_operands(insn.op_str)
    if operands:
        if insn.imm_size:
            reloc = reloc_by_operand_offset.get(insn.imm_offset)
            if reloc is not None:
                operands[-1] = relocation_target(
                    reloc,
                    labels,
                    for_relative=reloc.relocation_type == IMAGE_REL_I386_REL32,
                )
            elif insn.mnemonic.startswith("j"):
                target = insn.operands[-1].imm
                symbol_at_target = section.symbols.get(target - section.address)
                if symbol_at_target is not None:
                    operands[-1] = asm_name(text_symbol_name(symbol_at_target))
                else:
                    labels.add(target)
                    operands[-1] = address_label(target)
        if insn.disp_size:
            reloc = reloc_by_operand_offset.get(insn.disp_offset)
            if reloc is not None:
                label = relocation_target(reloc, labels, for_relative=False)
                for i, operand in enumerate(operands):
                    if "[" in operand and "]" in operand:
                        operands[i] = replace_memory_displacement(operand, label)
                        break

    if operands:
        return f"{insn.mnemonic} {', '.join(operands)}"
    return insn.mnemonic


def _next_data_boundary(
    offset: int, end: int, data_bases: Set[int], reloc_offsets: Set[int]
) -> int:
    """The end of a reloc-less data run starting at `offset`: the next data-table
    base, the next relocation (which starts a `.long` jump table), or the section
    end — whichever comes first."""
    limit = end
    for candidate in data_bases | reloc_offsets:
        if offset < candidate < limit:
            limit = candidate
    return limit


def decode_text_section(cap: cs.Cs, section: CoffSection) -> List[Element]:
    """Linearly decode a `.text` section into instructions and in-`.text` data.

    MSVC emits switch jump tables and 2-level case-mapping byte maps as data
    *inside* `.text`, after the function body. A naive linear sweep decodes these
    zero-filled/relocated bytes as garbage instructions. Two structural signals
    separate data from code without guessing:

    * A relocation the code stream lands on (i.e. at an instruction *start*) is a
      data word: a real x86 instruction begins with an opcode byte, never with
      its own operand's relocation, so any relocation we reach at a decode
      boundary is a 4-byte jump-table entry (`.long <label>`).
    * A reloc-less case-mapping byte map has no per-entry relocation, but its base
      is referenced by an absolute (DIR32) memory operand elsewhere in the same
      section (`mov dl, byte ptr [eax + $Lxxx]`). Direct branches within a
      section are PC-relative and *not* relocated, so an in-section absolute
      reference can only point at such a data table; those bytes are emitted raw.

    Every relocation must be either consumed as an instruction operand or emitted
    as a data word; a leftover relocation, or a byte capstone cannot decode,
    raises rather than silently baking wrong ground truth into a fixture.
    """
    elements: List[Element] = []
    data = section.data
    # MSVC pads a COMDAT `.text` section's tail with single-byte `nop` (0x90) to
    # the section's alignment. That padding is not code, and — crucially — when
    # it trails a 2-level switch's case-mapping byte map it would otherwise be
    # read as extra (out-of-range) map entries. No function body or table ends in
    # a 0x90 run (a byte map's real entries are small jump-table indices), so the
    # trailing 0x90 run is unambiguously padding and is dropped.
    end = len(data)
    while end > 0 and data[end - 1] == 0x90:
        end -= 1
    reloc_offsets = set(section.relocations.keys())
    data_bases: Set[int] = set()
    consumed_relocs: Set[int] = set()

    offset = 0
    while offset < end:
        if offset in reloc_offsets:
            reloc = section.relocations[offset]
            elements.append(DataLong(section.address + offset, reloc))
            consumed_relocs.add(offset)
            offset += 4
            continue

        if offset in data_bases:
            limit = _next_data_boundary(offset, end, data_bases, reloc_offsets)
            for i in range(offset, limit):
                elements.append(DataByte(section.address + i, data[i]))
            offset = limit
            continue

        insns = list(cap.disasm(data[offset:end], section.address + offset, count=1))
        if not insns:
            raise ValueError(
                f"{section.name}: cannot decode byte {data[offset]:#04x} at "
                f"offset {offset:#x}"
            )
        insn = insns[0]
        elements.append(Insn(insn, insn.address, insn.bytes))
        for reloc_offset in range(offset, offset + insn.size):
            reloc = section.relocations.get(reloc_offset)
            if reloc is None:
                continue
            consumed_relocs.add(reloc_offset)
            if (
                reloc.relocation_type in (IMAGE_REL_I386_DIR32, IMAGE_REL_I386_DIR32NB)
                and reloc.symbol.section is section
            ):
                data_bases.add(reloc.symbol.offset + reloc.symbol_offset)
        offset += insn.size

    dropped = reloc_offsets - consumed_relocs
    if dropped:
        raise ValueError(
            f"{section.name}: relocations dropped at offsets "
            + ", ".join(hex(o) for o in sorted(dropped))
        )
    return elements


def collect_relocation_labels(section: CoffSection, labels: Set[int]) -> None:
    for reloc in section.relocations.values():
        target_section = reloc.symbol.section
        if target_section is not None and target_section.name == ".text":
            target_offset = reloc.symbol.offset + reloc.symbol_offset
            if target_offset not in target_section.symbols:
                labels.add(target_section.address + target_offset)


def collect_jump_labels(
    elements: List[Element],
    section: CoffSection,
    labels: Set[int],
) -> None:
    for element in elements:
        if not isinstance(element, Insn):
            continue
        insn = element.insn
        if insn.mnemonic.startswith("j") and insn.operands and insn.imm_size:
            target = insn.operands[-1].imm
            if target - section.address not in section.symbols:
                labels.add(target)


def disassemble_text_section(
    section: CoffSection,
    elements: List[Element],
    labels: Set[int],
    output: TextIO,
) -> None:
    for element in elements:
        addr = element.address
        symbol = section.symbols.get(addr - section.address)
        if symbol is not None:
            if re.fullmatch(r"[_@][^@]+@\d+", symbol.name):
                output.write(f"# MSVC symbol: {asm_name(symbol.name)}\n")
            output.write(f"{asm_name(text_symbol_name(symbol))}:\n")
        if addr in labels:
            output.write(f"{address_label(addr)}:\n")
        if isinstance(element, Insn):
            prefix = "/* %08X %04X  %s */" % (
                addr,
                addr - section.address,
                " ".join(f"{b:02X}" for b in element.raw),
            )
            output.write(
                f"{prefix}\t{instruction_to_text(element.insn, section, labels)}\n"
            )
        elif isinstance(element, DataLong):
            output.write(f"\t.long {relocation_word(element.reloc, labels)}\n")
        else:
            output.write(f"\t.byte 0x{element.value:02X}\n")


def relocation_word(reloc: CoffRelocation, labels: Set[int]) -> str:
    if reloc.relocation_type in (IMAGE_REL_I386_DIR32, IMAGE_REL_I386_DIR32NB):
        return relocation_target(reloc, labels, for_relative=False)
    return relocation_target(reloc, labels, for_relative=True)


def disassemble_data_section(
    section: CoffSection,
    output: TextIO,
    real_consts: Optional[Dict[int, Tuple[str, str, int]]] = None,
) -> None:
    labels: Set[int] = set()
    collect_relocation_labels(section, labels)
    real_consts = real_consts or {}

    offset = 0
    symbols = sorted(sym for sym in section.symbols.values() if sym.name)
    symbol_offsets = {sym.offset for sym in symbols}
    total = max(len(section.data), section.size)

    while offset < total:
        symbol = section.symbols.get(offset)
        if symbol is not None:
            output.write(f"{asm_name(symbol_name(symbol))}:\n")

        if offset >= len(section.data):
            # Uninitialized (.bss) tail: reserve the run up to the next symbol
            # (or the section end) so the symbol keeps its size and m2c can
            # recover its type.
            following = [o for o in symbol_offsets if o > offset]
            limit = min(following) if following else total
            output.write(f"\t.space 0x{limit - offset:X}\n")
            offset = limit
            continue

        real = real_consts.get(offset)
        if real is not None:
            directive, literal, size = real
            output.write(f"\t{directive} {literal}\n")
            offset += size
            continue

        reloc = section.relocations.get(offset)
        if reloc is not None and offset + 4 <= len(section.data):
            output.write(f"\t.long {relocation_word(reloc, labels)}\n")
            offset += 4
            continue

        next_symbol_offsets = [
            sym_offset for sym_offset in symbol_offsets if sym_offset > offset
        ]
        limit = min(next_symbol_offsets) if next_symbol_offsets else len(section.data)
        next_reloc_offsets = [
            reloc_offset
            for reloc_offset in section.relocations.keys()
            if reloc_offset > offset
        ]
        if next_reloc_offsets:
            limit = min(limit, min(next_reloc_offsets))

        if offset + 4 <= limit and offset % 4 == 0:
            (word,) = struct.unpack("<I", section.data[offset : offset + 4])
            output.write(f"\t.long 0x{word:08X}\n")
            offset += 4
        else:
            byte = section.data[offset]
            output.write(f"\t.byte 0x{byte:02X}\n")
            offset += 1


def should_emit_section(section: CoffSection) -> bool:
    if not section.data and not section.symbols:
        return False
    if section.name.startswith(".debug") or section.name == ".drectve":
        return False
    return section.name in (".text", ".data", ".rdata", ".bss")


def disassemble_msvc_coff(coff_in: BinaryIO, asm_out: TextIO) -> None:
    coff = CoffFile.parse(coff_in.read())
    sections = [section for section in coff.sections if should_emit_section(section)]

    # Lay the sections out at distinct addresses. MSVC /Gy-style COMDATs give
    # every function its own `.text` section, all with virtual address 0;
    # without a layout pass, `.L<address>` labels would collide across
    # functions.
    address = 0
    for section in sections:
        section.address = address
        address += max(len(section.data), section.size)

    # Decode MSVC `__real@` float/double constants: rename their symbols to
    # clean identifiers (shared with relocations, so references follow) and
    # record which offsets to emit as `.float`/`.double`.
    real_consts = rename_real_constants(sections)

    cap = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_32)
    cap.detail = True
    cap.syntax = cs.CS_OPT_SYNTAX_INTEL

    # First pass: collect all label targets, so that labels referenced across
    # sections (e.g. jump tables) are known before any section is emitted.
    disassemblies: Dict[int, List[Element]] = {}
    labels: Set[int] = set()
    for i, section in enumerate(sections):
        collect_relocation_labels(section, labels)
        if section.name == ".text":
            disassembly = decode_text_section(cap, section)
            disassemblies[i] = disassembly
            collect_jump_labels(disassembly, section, labels)

    previous_name: Optional[str] = None
    for i, section in enumerate(sections):
        if i != 0:
            asm_out.write("\n")
        if section.name != previous_name:
            asm_out.write(f".section {section.name}\n")
            previous_name = section.name
        if section.name == ".text":
            disassemble_text_section(section, disassemblies[i], labels, asm_out)
        else:
            disassemble_data_section(section, asm_out, real_consts.get(i))


if __name__ == "__main__":
    disassemble_msvc_coff(sys.stdin.buffer, sys.stdout)
