from __future__ import annotations
import re
import struct
import sys
from typing import BinaryIO, Dict, List, Optional, Set, TextIO, Tuple

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


RE_VALID_UNQUOTED = re.compile(r"^[A-Za-z0-9_.$]+$")


def address_label(addr: int) -> str:
    return f".L{addr:08X}"


def symbol_name(sym: CoffSymbol) -> str:
    return sym.name


def text_symbol_name(sym: CoffSymbol) -> str:
    if sym.section is not None and sym.section.name == ".text":
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


def replace_memory_displacement(operand: str, label: str) -> str:
    start = operand.index("[")
    end = operand.rindex("]")
    inner = operand[start + 1 : end].strip()
    if inner in ("0", "0x0"):
        new_inner = label
    elif re.search(r"([+-]\s*)?0x?0$", inner):
        new_inner = re.sub(r"\s*[+]\s*0x?0$", f" + {label}", inner)
        new_inner = re.sub(r"\s*-\s*0x?0$", f" + {label}", new_inner)
        if new_inner == inner:
            new_inner = re.sub(r"0x?0$", label, inner)
    else:
        new_inner = f"{inner} + {label}"
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


def disassemble_bytes(
    cap: cs.Cs, base_address: int, data: bytes
) -> List[Tuple[cs.CsInsn, int, bytes]]:
    output: List[Tuple[cs.CsInsn, int, bytes]] = []
    offset = 0
    end = len(data)
    while offset < end:
        code = data[offset:end]
        insns = list(cap.disasm(code, base_address + offset, count=1))
        if not insns:
            break
        insn = insns[0]
        output.append((insn, insn.address, insn.bytes))
        offset += insn.size
    return output


def collect_relocation_labels(section: CoffSection, labels: Set[int]) -> None:
    for reloc in section.relocations.values():
        target_section = reloc.symbol.section
        if target_section is not None and target_section.name == ".text":
            target_offset = reloc.symbol.offset + reloc.symbol_offset
            if target_offset not in target_section.symbols:
                labels.add(target_section.address + target_offset)


def collect_jump_labels(
    disassembly: List[Tuple[cs.CsInsn, int, bytes]],
    section: CoffSection,
    labels: Set[int],
) -> None:
    for insn, _addr, _raw in disassembly:
        if insn.mnemonic.startswith("j") and insn.operands and insn.imm_size:
            target = insn.operands[-1].imm
            if target - section.address not in section.symbols:
                labels.add(target)


def disassemble_text_section(
    section: CoffSection,
    disassembly: List[Tuple[cs.CsInsn, int, bytes]],
    labels: Set[int],
    output: TextIO,
) -> None:
    for insn, addr, raw in disassembly:
        symbol = section.symbols.get(addr - section.address)
        if symbol is not None:
            output.write(f"{asm_name(text_symbol_name(symbol))}:\n")
        if addr in labels:
            output.write(f"{address_label(addr)}:\n")
        prefix = "/* %08X %04X  %s */" % (
            addr,
            addr - section.address,
            " ".join(f"{b:02X}" for b in raw),
        )
        output.write(f"{prefix}\t{instruction_to_text(insn, section, labels)}\n")


def relocation_word(reloc: CoffRelocation, labels: Set[int]) -> str:
    if reloc.relocation_type in (IMAGE_REL_I386_DIR32, IMAGE_REL_I386_DIR32NB):
        return relocation_target(reloc, labels, for_relative=False)
    return relocation_target(reloc, labels, for_relative=True)


def disassemble_data_section(section: CoffSection, output: TextIO) -> None:
    labels: Set[int] = set()
    collect_relocation_labels(section, labels)

    offset = 0
    symbols = sorted(sym for sym in section.symbols.values() if sym.name)
    symbol_offsets = {sym.offset for sym in symbols}

    while offset < len(section.data):
        symbol = section.symbols.get(offset)
        if symbol is not None:
            output.write(f"{asm_name(symbol_name(symbol))}:\n")

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
        address += len(section.data)

    cap = cs.Cs(cs.CS_ARCH_X86, cs.CS_MODE_32)
    cap.detail = True
    cap.syntax = cs.CS_OPT_SYNTAX_INTEL

    # First pass: collect all label targets, so that labels referenced across
    # sections (e.g. jump tables) are known before any section is emitted.
    disassemblies: Dict[int, List[Tuple[cs.CsInsn, int, bytes]]] = {}
    labels: Set[int] = set()
    for i, section in enumerate(sections):
        collect_relocation_labels(section, labels)
        if section.name == ".text":
            disassembly = disassemble_bytes(cap, section.address, section.data)
            disassemblies[i] = disassembly
            collect_jump_labels(disassembly, section, labels)

    previous_name: Optional[str] = None
    for i, section in enumerate(sections):
        if section.name != previous_name:
            asm_out.write(f".section {section.name}\n")
            previous_name = section.name
        if section.name == ".text":
            disassemble_text_section(section, disassemblies[i], labels, asm_out)
        else:
            disassemble_data_section(section, asm_out)
        asm_out.write("\n")


if __name__ == "__main__":
    disassemble_msvc_coff(sys.stdin.buffer, sys.stdout)
