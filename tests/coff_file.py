from __future__ import annotations
import struct
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


IMAGE_FILE_MACHINE_I386 = 0x014C
IMAGE_SYM_UNDEFINED = 0
IMAGE_SYM_ABSOLUTE = -1
IMAGE_SYM_DEBUG = -2

IMAGE_REL_I386_DIR32 = 0x0006
IMAGE_REL_I386_DIR32NB = 0x0007
IMAGE_REL_I386_SECREL = 0x000B
IMAGE_REL_I386_REL32 = 0x0014


@dataclass(order=True)
class CoffSymbol:
    offset: int
    name: str
    section: Optional[CoffSection]


@dataclass(order=True)
class CoffRelocation:
    section_offset: int
    symbol: CoffSymbol
    symbol_offset: int
    relocation_type: int


@dataclass
class CoffSection:
    address: int
    name: str
    data: bytes = field(repr=False)
    relocations: Dict[int, CoffRelocation] = field(default_factory=dict, repr=False)
    symbols: Dict[int, CoffSymbol] = field(default_factory=dict, repr=False)


@dataclass
class CoffFile:
    sections: Dict[str, CoffSection] = field(default_factory=dict)
    symbols: Dict[str, CoffSymbol] = field(default_factory=dict)

    @staticmethod
    def parse(data: bytes) -> CoffFile:
        if len(data) < 20:
            raise ValueError("Input data is too small to be a COFF object")

        def read(spec: str, offset: int) -> Tuple[int, ...]:
            size = struct.calcsize("<" + spec)
            return struct.unpack("<" + spec, data[offset : offset + size])

        def read_c_string(offset: int) -> str:
            end = data.index(b"\0", offset)
            return data[offset:end].decode("latin1")

        def read_symbol_name(raw: bytes, strtab_offset: int) -> str:
            zeroes, str_offset = struct.unpack("<II", raw)
            if zeroes == 0:
                return read_c_string(strtab_offset + str_offset)
            return raw.rstrip(b"\0").decode("latin1")

        (
            machine,
            section_count,
            _timestamp,
            symbol_table_offset,
            symbol_count,
            optional_header_size,
            _characteristics,
        ) = read("HHIIIHH", 0)
        if machine != IMAGE_FILE_MACHINE_I386:
            raise ValueError(f"Input COFF is not i386 (machine = {machine:#x})")

        section_headers_offset = 20 + optional_header_size
        strtab_offset = symbol_table_offset + symbol_count * 18
        coff = CoffFile()

        @dataclass
        class SectionHeader:
            name: str
            virtual_size: int
            virtual_address: int
            size_of_raw_data: int
            pointer_to_raw_data: int
            pointer_to_relocations: int
            pointer_to_linenumbers: int
            number_of_relocations: int
            number_of_linenumbers: int
            characteristics: int

        section_headers: List[SectionHeader] = []
        for i in range(section_count):
            offset = section_headers_offset + i * 40
            raw_name = data[offset : offset + 8]
            (
                virtual_size,
                virtual_address,
                size_of_raw_data,
                pointer_to_raw_data,
                pointer_to_relocations,
                pointer_to_linenumbers,
                number_of_relocations,
                number_of_linenumbers,
                characteristics,
            ) = read("IIIIIIHHI", offset + 8)
            name = raw_name.rstrip(b"\0").decode("latin1")
            section_headers.append(
                SectionHeader(
                    name=name,
                    virtual_size=virtual_size,
                    virtual_address=virtual_address,
                    size_of_raw_data=size_of_raw_data,
                    pointer_to_raw_data=pointer_to_raw_data,
                    pointer_to_relocations=pointer_to_relocations,
                    pointer_to_linenumbers=pointer_to_linenumbers,
                    number_of_relocations=number_of_relocations,
                    number_of_linenumbers=number_of_linenumbers,
                    characteristics=characteristics,
                )
            )
            section_data = data[
                pointer_to_raw_data : pointer_to_raw_data + size_of_raw_data
            ]
            coff.sections[name] = CoffSection(
                address=virtual_address, name=name, data=section_data
            )

        sections_by_index: Dict[int, CoffSection] = {
            i + 1: coff.sections[header.name]
            for i, header in enumerate(section_headers)
            if header.name in coff.sections
        }

        symbols_by_index: List[CoffSymbol] = []
        i = 0
        while i < symbol_count:
            offset = symbol_table_offset + i * 18
            name = read_symbol_name(data[offset : offset + 8], strtab_offset)
            value, section_number, _typ, _storage_class, aux_count = struct.unpack(
                "<IhHBB", data[offset + 8 : offset + 18]
            )
            if section_number in (
                IMAGE_SYM_UNDEFINED,
                IMAGE_SYM_ABSOLUTE,
                IMAGE_SYM_DEBUG,
            ):
                section = None
            else:
                section = sections_by_index.get(section_number)
            symbol = CoffSymbol(offset=value, name=name, section=section)
            coff.symbols[name] = symbol
            symbols_by_index.append(symbol)
            for _ in range(aux_count):
                symbols_by_index.append(symbol)
            i += 1 + aux_count

            if (
                section is not None
                and name
                and name not in coff.sections
                and value not in section.symbols
            ):
                section.symbols[value] = symbol

        for header in section_headers:
            section = coff.sections[header.name]
            for j in range(header.number_of_relocations):
                offset = header.pointer_to_relocations + j * 10
                virtual_address, symbol_index, relocation_type = struct.unpack(
                    "<IIH", data[offset : offset + 10]
                )
                symbol = symbols_by_index[symbol_index]
                symbol_offset = 0
                if relocation_type in (
                    IMAGE_REL_I386_DIR32,
                    IMAGE_REL_I386_DIR32NB,
                    IMAGE_REL_I386_SECREL,
                    IMAGE_REL_I386_REL32,
                ):
                    (symbol_offset,) = read(
                        "i", header.pointer_to_raw_data + virtual_address
                    )
                reloc = CoffRelocation(
                    section_offset=virtual_address,
                    symbol=symbol,
                    symbol_offset=symbol_offset,
                    relocation_type=relocation_type,
                )
                section.relocations[virtual_address] = reloc

        return coff
