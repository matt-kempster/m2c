#!/usr/bin/env python3
import argparse
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import typing
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Mapping, NamedTuple, Optional, Tuple

OUT_FILES_TO_IRIX_FLAG: Mapping[str, str] = {"irix-g": "-g", "irix-o2": "-O2"}
QEMU_IRIX: Optional[str] = None
IRIX_ROOT: Optional[str] = None
SM64_TOOLS: Optional[str] = None


def set_up_logging(debug: bool) -> None:
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )


def get_environment_variables() -> bool:
    global QEMU_IRIX, IRIX_ROOT, SM64_TOOLS
    success = True
    try:
        QEMU_IRIX = str(os.environ["QEMU_IRIX"])
    except KeyError:
        logging.error("env variable QEMU_IRIX should point to the qemu-mips binary")
        success = False
    try:
        IRIX_ROOT = str(os.environ["IRIX_ROOT"])
    except KeyError:
        logging.error(
            "env variable IRIX_ROOT should point to the IRIX compiler directory"
        )
        success = False
    try:
        SM64_TOOLS = str(os.environ["SM64_TOOLS"])
    except KeyError:
        logging.error(
            "env variable SM64_TOOLS should point to a checkout of "
            "https://github.com/queueRAM/sm64tools/, with mipsdisasm built"
        )
        success = False
    return success


class DisassemblyInfo(NamedTuple):
    entry: bytes
    disasm: bytes


def do_disassembly_step(temp_out_file: str) -> DisassemblyInfo:
    section_lines: List[str] = subprocess.run(
        ["mips-linux-gnu-readelf", "-S", temp_out_file],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    ).stdout.split("\n")

    header_lines: List[bytes] = subprocess.run(
        ["mips-linux-gnu-readelf", "-h", temp_out_file], stdout=subprocess.PIPE
    ).stdout.split(b"\n")

    for line in header_lines:
        if b"Entry" in line:
            entry = line.split(b" ")[-1]
            break

    for line in section_lines:
        if " .text" not in line:
            continue
        addr = "0x" + line[42:][:7]
        index = "0x" + line[51:][:5]
        size = "0x" + line[58:][:5]
        break

    arg = f"{addr}:{index}+{size}"
    logging.debug(f"Calling mipsdisasm with arg {arg} and entry point {entry}...")
    final_asm = subprocess.run(
        [str(SM64_TOOLS) + "/mipsdisasm", temp_out_file, arg], stdout=subprocess.PIPE
    ).stdout

    if final_asm is None:
        raise Exception("mipsdisasm didn't output anything")

    return DisassemblyInfo(entry, final_asm)


def do_linker_step(temp_out_file: str, temp_o_file: str) -> None:
    subprocess.run(
        ["mips-linux-gnu-ld", "-o", temp_out_file, temp_o_file, "-e", "test"]
    )


def do_compilation_step(temp_o_file: str, in_file: str, flag: str) -> None:
    subprocess.run(
        [
            str(QEMU_IRIX),
            "-silent",
            "-L",
            str(IRIX_ROOT),
            str(IRIX_ROOT) + "/usr/bin/cc",
            "-c",
            "-Wab,-r4300_mul",
            "-non_shared",
            "-G",
            "0",
            "-Xcpluscomm",
            "-fullwarn",
            "-wlint",
            "-woff",
            "819,820,852,821",
            "-signed",
            "-DVERSION_JP=1",
            "-mips2",
            "-o",
            temp_o_file,
            in_file,
            flag,
        ]
    )


def do_fix_lohi_step(disasm_info: DisassemblyInfo) -> bytes:
    logging.debug("Fixing %lo and %hi macros...")

    def replace_last(s: bytes, a: bytes, b: bytes) -> bytes:
        return b.join(s.rsplit(a, 1))

    output: List[bytes] = []
    waiting_hi: Dict[bytes, Tuple[int, int, bytes]] = {}
    for line in disasm_info.disasm.split(b"\n"):
        line = line.strip()
        if (disasm_info.entry + b" ")[2:].upper() in line:
            output.append(b"glabel test")

        index = len(output)
        if line.startswith(b"func"):
            line = b"glabel " + line[:-1]
        elif b"lui" in line:
            reg, imm_hex = line.split(b" ")[-2::]
            imm = int(imm_hex, 0)
            if 0x10 <= imm < 0x1000:  # modestly large lui is probably a pointer
                waiting_hi[reg[:-1]] = (index, imm, imm_hex)
        else:
            lo_regs = [reg for reg in waiting_hi.keys() if reg in line]
            if lo_regs and "0x" in line:
                lo_reg = lo_regs[0]
                hi_ind, hi_imm, hi_imm_hex = waiting_hi[lo_reg]
                lo_imm_hex = line.split(b" ")[-1].split(b"(")[0]
                lo_imm = int(lo_imm_hex, 0)
                if lo_imm >= 0x8000:
                    lo_imm -= 0x10000
                sym = b"D_" + bytes(hex((hi_imm << 16) + lo_imm)[2:].upper(), "utf-8")
                line = replace_last(line, lo_imm_hex, b"%lo(" + sym + b")")
                output[hi_ind] = replace_last(
                    output[hi_ind], hi_imm_hex, b"%hi(" + sym + b")"
                )
                del waiting_hi[lo_reg]
        output.append(line)
    return b"\n".join(output)


def irix_compile_with_flag(in_file: Path, out_file: Path, flag: str) -> None:
    logging.info(f"Compiling {in_file} to {out_file} using this flag: {flag}")
    with ExitStack() as stack:
        temp_o_file = stack.enter_context(NamedTemporaryFile(suffix=".o")).name
        temp_out_file = stack.enter_context(NamedTemporaryFile(suffix=".out")).name
        logging.debug(f"Compiling and linking {in_file} with {flag}...")
        do_compilation_step(temp_o_file, str(in_file), flag)
        do_linker_step(temp_out_file, temp_o_file)
        disasm_info = do_disassembly_step(temp_out_file)
        final_asm = do_fix_lohi_step(disasm_info)
    out_file.write_bytes(final_asm)
    logging.info(f"Successfully wrote disassembly to {out_file}.")


def add_test_from_file(orig_file: Path) -> None:
    test_dir = Path("end_to_end") / orig_file.stem
    try:
        test_dir.mkdir()
    except FileExistsError:
        raise Exception(f"{test_dir} already exists. Name your test something else.")
    logging.debug(f"Created new directory: {test_dir}")

    in_file: Path = test_dir / "orig.c"
    shutil.copy(str(orig_file), str(in_file))
    logging.debug(f"Created {in_file}")

    for asm_filename, flag in OUT_FILES_TO_IRIX_FLAG.items():
        asm_file_path = Path(str(test_dir / asm_filename) + ".s")
        irix_compile_with_flag(in_file, asm_file_path, flag)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add or update end-to-end decompiler tests."
    )
    parser.add_argument(
        "files", help="files containing C code to compile (then decompile)", nargs="+"
    )
    parser.add_argument(
        "--debug", dest="debug", help="print debug info", action="store_true"
    )

    args = parser.parse_args()
    set_up_logging(args.debug)

    if not get_environment_variables():
        return 2

    for orig_file in args.files:
        if not Path(orig_file).is_file():
            logging.error(f"{orig_file} does not exist. Skipping.")
            continue
        add_test_from_file(Path(orig_file))

    return 0


if __name__ == "__main__":
    sys.exit(main())
