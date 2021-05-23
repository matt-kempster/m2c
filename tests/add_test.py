#!/usr/bin/env python3
import argparse
import distutils.spawn
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

COMMON_IRIX_FLAGS = ["-woff", "826"]

OUT_FILES_TO_IRIX_FLAGS: Mapping[str, List[str]] = {
    "irix-g": ["-g", "-mips2"],
    "irix-o2": ["-O2", "-mips2"],
    # "irix-g-mips1": ["-g", "-mips1"],
    # "irix-o2-mips1": ["-O2", "-mips1"],
}


def set_up_logging(debug: bool) -> None:
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )


class PathsToBinaries(NamedTuple):
    MIPS_CC: str
    SM64_TOOLS: str


def get_environment_variables() -> Optional[PathsToBinaries]:
    def load(env_var_name: str, error_message: str) -> Optional[str]:
        env_var = os.environ.get(env_var_name)
        if env_var is None:
            logging.error(error_message)
        return env_var

    MIPS_CC = load(
        "MIPS_CC",
        "env variable MIPS_CC should point to recompiled IDO cc binary",
    )
    SM64_TOOLS = load(
        "SM64_TOOLS",
        (
            "env variable SM64_TOOLS should point to a checkout of "
            "https://github.com/queueRAM/sm64tools/, with mipsdisasm built"
        ),
    )
    if not SM64_TOOLS or not MIPS_CC:
        logging.error(
            "One or more required environment variables are not set. Bailing."
        )
        return None
    else:
        return PathsToBinaries(MIPS_CC, SM64_TOOLS)


class DisassemblyInfo(NamedTuple):
    entry_point: bytes
    disasm: bytes


def do_disassembly_step(
    temp_out_file: str, env_vars: PathsToBinaries
) -> DisassemblyInfo:
    section_lines: List[str] = subprocess.run(
        ["mips-linux-gnu-readelf", "-S", temp_out_file],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    ).stdout.split("\n")

    header_lines: List[bytes] = subprocess.run(
        ["mips-linux-gnu-readelf", "-h", temp_out_file], stdout=subprocess.PIPE
    ).stdout.split(b"\n")

    entry_point: bytes = b""
    for line in header_lines:
        if b"Entry" in line:
            entry_point = line.split(b" ")[-1]
            break
    if not entry_point:
        raise Exception("no entry point found in ELF file")

    for section_line in section_lines:
        if " .text" not in section_line:
            continue
        addr = "0x" + section_line[42:][:7]
        index = "0x" + section_line[51:][:5]
        size = "0x" + section_line[58:][:5]
        break

    arg = f"{addr}:{index}+{size}"
    entry_point_str = entry_point.decode("utf-8", "replace")
    logging.debug(
        f"Calling mipsdisasm with arg {arg} and entry point {entry_point_str}..."
    )
    final_asm = subprocess.run(
        [env_vars.SM64_TOOLS + "/mipsdisasm", temp_out_file, arg],
        stdout=subprocess.PIPE,
    ).stdout

    if final_asm is None:
        raise Exception("mipsdisasm didn't output anything")

    return DisassemblyInfo(entry_point, final_asm)


def do_linker_step(temp_out_file: str, temp_o_file: str) -> None:
    subprocess.run(
        ["mips-linux-gnu-ld", "-o", temp_out_file, temp_o_file, "-e", "test"]
    )


def do_compilation_step(
    temp_o_file: str, in_file: str, flags: List[str], env_vars: PathsToBinaries
) -> None:
    subprocess.run(
        [
            env_vars.MIPS_CC,
            "-c",
            "-Wab,-r4300_mul",
            "-non_shared",
            "-G",
            "0",
            "-Xcpluscomm",
            "-fullwarn",
            "-wlint",
            "-woff",
            "819,820,852,821,827",
            "-signed",
            "-o",
            temp_o_file,
            in_file,
            *flags,
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
        if (disasm_info.entry_point + b" ")[2:].upper() in line:
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
            if lo_regs and b"0x" in line:
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


def irix_compile_with_flag(
    in_file: Path, out_file: Path, flags: List[str], env_vars: PathsToBinaries
) -> None:
    flags_str = " ".join(flags)
    logging.info(f"Compiling {in_file} to {out_file} using these flags: {flags_str}")
    with ExitStack() as stack:
        temp_o_file = stack.enter_context(NamedTemporaryFile(suffix=".o")).name
        temp_out_file = stack.enter_context(NamedTemporaryFile(suffix=".out")).name
        logging.debug(f"Compiling and linking {in_file} with {flags_str}...")
        do_compilation_step(temp_o_file, str(in_file), flags, env_vars)
        do_linker_step(temp_out_file, temp_o_file)
        disasm_info = do_disassembly_step(temp_out_file, env_vars)
        final_asm = do_fix_lohi_step(disasm_info)
    out_file.write_bytes(final_asm)
    logging.info(f"Successfully wrote disassembly to {out_file}.")


def add_test_from_file(orig_file: Path, env_vars: PathsToBinaries) -> None:
    test_dir = orig_file.parent
    for asm_filename, flags in OUT_FILES_TO_IRIX_FLAGS.items():
        asm_file_path = Path(str(test_dir / asm_filename) + ".s")
        irix_compile_with_flag(
            orig_file, asm_file_path, flags + COMMON_IRIX_FLAGS, env_vars
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add or update end-to-end decompiler tests."
    )
    parser.add_argument(
        "files",
        help=(
            "Files containing C code to compile (then decompile). "
            "Each one must have a path of the form "
            "`tests/end_to_end/TEST_NAME/orig.c`."
        ),
        nargs="+",
    )
    parser.add_argument(
        "--debug", dest="debug", help="print debug info", action="store_true"
    )

    args = parser.parse_args()
    set_up_logging(args.debug)

    env_vars = get_environment_variables()
    if env_vars is None:
        return 2

    for orig_filename in args.files:
        orig_file = Path(orig_filename)
        if not orig_file.is_file():
            logging.error(f"{orig_file} does not exist. Skipping.")
            continue
        expected_file = (
            Path(__file__).parent / "end_to_end" / orig_file.parent.name / "orig.c"
        )
        if orig_file != expected_file:
            logging.error(
                f"`{orig_file}` does not have a path of the form `{expected_file}`! Skipping."
            )
            continue
        add_test_from_file(orig_file, env_vars)

    return 0


if __name__ == "__main__":
    sys.exit(main())
