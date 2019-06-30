#!/usr/bin/env python3
import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path

from src.main import main as decompile
from src.options import Options

CRASH_STRING = "CRASHED\n"


def set_up_logging(debug: bool) -> None:
    logging.basicConfig(
        format="[%(levelname)s] %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )


def decompile_and_compare(
    asm_file_path: Path, output_path: Path, should_overwrite: bool
) -> bool:
    logging.debug(
        f"Decompiling {asm_file_path}"
        + (f" into {output_path}" if should_overwrite else "")
    )
    try:
        original_contents = output_path.read_text()
    except FileNotFoundError:
        logging.info(f"{output_path} does not exist. Creating...")
        original_contents = "(file did not exist)"

    final_contents = decompile_and_capture_output(output_path, asm_file_path)

    if should_overwrite:
        output_path.write_text(final_contents)

    changed = final_contents != original_contents
    if changed:
        logging.info(
            "\n".join(
                [
                    f"Output of {asm_file_path} changed!",
                    "PREVIOUS:",
                    original_contents,
                    "NEW:",
                    final_contents,
                ]
            )
        )
    return changed


def decompile_and_capture_output(output_path: Path, asm_file_path: Path) -> str:
    out_string = io.StringIO()
    with contextlib.redirect_stdout(out_string):
        returncode = decompile(
            Options(
                filename=str(asm_file_path),
                debug=False,
                void=False,
                ifs=True,
                andor_detection=True,
                goto_patterns=["GOTO"],
                rodata_files=[],
                stop_on_error=True,
                print_assembly=False,
                visualize_flowgraph=False,
                preproc_defines={},
            ),
            "test",
        )
    if returncode == 0:
        return out_string.getvalue()
    else:
        return CRASH_STRING


def run_e2e_test(e2e_test_path: Path, should_overwrite: bool) -> bool:
    logging.info(f"Running test: {e2e_test_path.name}")
    any_changed = False
    for asm_file_path in e2e_test_path.glob("*.s"):
        old_output_path = asm_file_path.parent.joinpath(asm_file_path.stem + "-out.c")
        any_changed = (
            decompile_and_compare(asm_file_path, old_output_path, should_overwrite)
            or any_changed
        )
    return any_changed


def main(should_overwrite: bool) -> int:
    any_changed = False
    for e2e_test_path in Path("tests/end_to_end").iterdir():
        any_changed = run_e2e_test(e2e_test_path, should_overwrite) or any_changed

    if any_changed:
        logging.info("The output of some tests changed.")
        return 1
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
