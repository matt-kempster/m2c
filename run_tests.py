#!/usr/bin/env python3
import argparse
import contextlib
import difflib
import io
import logging
import shlex
import sys
from pathlib import Path
from typing import List

from src.main import parse_flags
from src.main import run as decompile
from src.options import CodingStyle, Options

CRASH_STRING = "CRASHED\n"


def set_up_logging(debug: bool) -> None:
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )


def get_test_flags(flags_path: Path) -> List[str]:
    if not flags_path.is_file():
        return []

    flags_str = flags_path.read_text()
    flags_list = shlex.split(flags_str)
    try:
        context_index = flags_list.index("--context")
        relative_context_path: str = flags_list[context_index + 1]
        absolute_context_path: Path = flags_path.parent / relative_context_path
        flags_list[context_index + 1] = str(absolute_context_path)
    except ValueError:
        pass  # doesn't have --context flag
    except IndexError:
        raise Exception(f"{flags_path} contains --context without argument")

    return flags_list


def decompile_and_compare(
    asm_file_path: Path, output_path: Path, flags_path: Path, should_overwrite: bool
) -> None:
    logging.debug(
        f"Decompiling {asm_file_path}"
        + (f" into {output_path}" if should_overwrite else "")
    )
    try:
        original_contents = output_path.read_text()
    except FileNotFoundError:
        logging.info(f"{output_path} does not exist. Creating...")
        original_contents = "(file did not exist)"

    flags = [str(asm_file_path), "test", "--allman", "--stop-on-error"]
    flags_list = get_test_flags(flags_path)
    flags.extend(flags_list)

    options = parse_flags(flags)
    final_contents = decompile_and_capture_output(options)

    if should_overwrite:
        output_path.write_text(final_contents)

    changed = final_contents != original_contents
    if changed:
        logging.info(
            "\n".join(
                [
                    f"Output of {asm_file_path} changed! Diff:",
                    *difflib.unified_diff(
                        original_contents.splitlines(), final_contents.splitlines()
                    ),
                ]
            )
        )


def decompile_and_capture_output(options: Options) -> str:
    out_string = io.StringIO()
    with contextlib.redirect_stdout(out_string):
        returncode = decompile(options)
    if returncode == 0:
        return out_string.getvalue()
    else:
        return CRASH_STRING


def run_e2e_test(e2e_test_path: Path, should_overwrite: bool) -> None:
    logging.info(f"Running test: {e2e_test_path.name}")

    for asm_file_path in e2e_test_path.glob("*.s"):
        old_output_path = asm_file_path.parent.joinpath(asm_file_path.stem + "-out.c")
        flags_path = asm_file_path.parent.joinpath(asm_file_path.stem + "-flags.txt")
        decompile_and_compare(
            asm_file_path, old_output_path, flags_path, should_overwrite
        )


def main(should_overwrite: bool) -> int:
    for e2e_test_path in (Path(__file__).parent / "tests" / "end_to_end").iterdir():
        run_e2e_test(e2e_test_path, should_overwrite)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run and record end-to-end decompiler tests."
    )
    parser.add_argument(
        "--debug", dest="debug", help="print debug info", action="store_true"
    )
    parser.add_argument(
        "--overwrite",
        dest="should_overwrite",
        action="store_true",
        help=(
            "overwrite the contents of the test output files. "
            "Do this once before committing."
        ),
    )
    args = parser.parse_args()
    set_up_logging(args.debug)

    if args.should_overwrite:
        logging.info("Overwriting test output files.")
    sys.exit(main(args.should_overwrite))
