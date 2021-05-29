#!/usr/bin/env python3
import argparse
import contextlib
import difflib
import io
import logging
import re
import shlex
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

from src.main import parse_flags
from src.main import run as decompile
from src.options import Options

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
        raise Exception(f"{flags_path} contains --context without argument") from None

    return flags_list


def decompile_and_compare(
    asm_file_path: Path,
    output_path: Path,
    should_overwrite: bool = False,
    brief_crashes: bool = True,
    flags_path: Optional[Path] = None,
    flags: Optional[List[str]] = None,
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

    test_flags = ["--stop-on-error", str(asm_file_path)]
    if flags is not None:
        test_flags.extend(flags)
    if flags_path is not None:
        test_flags.extend(get_test_flags(flags_path))
    options = parse_flags(test_flags)

    final_contents = decompile_and_capture_output(options, brief_crashes)

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
    return should_overwrite or not changed


def decompile_and_capture_output(options: Options, brief_crashes: bool) -> str:
    out_string = io.StringIO()
    with contextlib.redirect_stdout(out_string):
        returncode = decompile(options)
    out_text = out_string.getvalue()
    # Rewrite paths in the output to be relative (e.g. in tracebacks)
    out_text = out_text.replace(str(Path(__file__).parent), ".")
    if returncode == 0:
        return out_text
    else:
        if brief_crashes:
            return CRASH_STRING
        else:
            return f"{CRASH_STRING}\n{out_text}"


def run_e2e_test(
    e2e_top_dir: Path,
    e2e_test_path: Path,
    should_overwrite: bool,
    filter_regex: Optional[str],
    coverage: Any,
) -> bool:

    ret = True
    for asm_file_path in e2e_test_path.glob("*.s"):
        old_output_path = asm_file_path.parent.joinpath(asm_file_path.stem + "-out.c")
        flags_path = asm_file_path.parent.joinpath(asm_file_path.stem + "-flags.txt")

        name = f"e2e:{asm_file_path.relative_to(e2e_top_dir)}"
        if filter_regex is not None and not re.search(filter_regex, name):
            continue
        if coverage:
            coverage.switch_context(name)
        logging.info(f"Running test: {name}")

        if not decompile_and_compare(
            asm_file_path,
            old_output_path,
            brief_crashes=True,
            should_overwrite=should_overwrite,
            flags_path=flags_path,
            flags=["test"],
        ):
            ret = False
    return ret


def run_project_tests(
    base_dir: Path,
    output_dir: Path,
    context_file: Optional[Path],
    should_overwrite: bool,
    filter_regex: Optional[str],
    name_prefix: str,
    coverage: Any,
) -> bool:
    ret = True
    asm_dir = base_dir / "asm"

    for asm_file in asm_dir.rglob("*"):
        if asm_file.suffix not in (".asm", ".s"):
            continue
        if "non_matching" in str(asm_file):
            continue

        asm_name = asm_file.name
        if (
            asm_name.startswith("code_data")
            or asm_name.startswith("code_rodata")
            or asm_name.startswith("boot_data")
            or asm_name.startswith("boot_rodata")
            or asm_name.endswith("_data.asm")
            or asm_name.endswith("_rodata.asm")
        ):
            continue

        flags = []
        if context_file is not None:
            flags.extend(["--context", str(context_file)])

        # Guess the name of .rodata file(s) for the MM decomp project
        for candidate in [
            # code/*.asm
            "code_rodata_" + asm_name,
            asm_name.replace("code_", "code_rodata_"),
            # boot/*.asm
            "boot_rodata_" + asm_name,
            asm_name.replace("boot_", "boot_rodata_"),
            # overlays/*.asm
            asm_name.rpartition("_0x")[0] + "_rodata.asm",
            asm_name.rpartition("_0x")[0] + "_late_rodata.asm",
        ]:
            if candidate == asm_name:
                continue
            f = asm_file.parent / candidate
            if f.exists():
                flags.extend(["--rodata", str(f)])

        test_path = asm_file.relative_to(asm_dir)
        name = f"{name_prefix}:{test_path}"
        if filter_regex is not None and not re.search(filter_regex, name):
            continue
        if coverage:
            coverage.switch_context(name)
        logging.info(f"Running test: {name}")

        output_file = (output_dir / test_path).with_suffix(".c")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if not decompile_and_compare(
            asm_file,
            output_file,
            brief_crashes=False,
            flags=flags,
            should_overwrite=should_overwrite,
        ):
            ret = False
    return ret


def main(
    project_dirs: List[Tuple[Path, bool]],
    should_overwrite: bool,
    filter_regex: Optional[str],
    coverage: Any,
) -> int:
    ret = 0
    e2e_top_dir = Path(__file__).parent / "tests" / "end_to_end"
    for e2e_test_path in e2e_top_dir.iterdir():
        if not run_e2e_test(
            e2e_top_dir, e2e_test_path, should_overwrite, filter_regex, coverage
        ):
            ret = 1

    for project_dir, use_context in project_dirs:
        name = project_dir.name
        context_file: Optional[Path] = None

        if use_context:
            name = f"{name}_ctx"
            context_file = project_dir / "ctx.c"
            if not context_file.exists():
                logging.error(
                    f"{project_dir} tests require context file, but {context_file} does not exist"
                )
                ret = 1
                continue

        output_dir = Path(__file__).parent / "tests" / "project" / name
        if not run_project_tests(
            project_dir,
            output_dir,
            context_file,
            should_overwrite,
            filter_regex,
            name,
            coverage,
        ):
            ret = 1

    return ret


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
    parser.add_argument(
        "--filter",
        dest="filter",
        help=("Only run tests matching this regular expression."),
    )
    parser.add_argument(
        "--project",
        dest="project_dirs",
        action="append",
        default=[],
        type=lambda p: (Path(p), False),
        help=(
            "Run tests on the asm files from a decompilation project. "
            "The zeldaret/oot and zeldaret/mm projects are supported. "
            "Can be specified multiple times."
        ),
    )
    parser.add_argument(
        "--project-with-context",
        dest="project_dirs",
        action="append",
        default=[],
        type=lambda p: (Path(p), True),
        help=(
            "Same as --project, but use the C context file `ctx.c` "
            "from the base directory. "
            "Can be specified multiple times."
        ),
    )
    args = parser.parse_args()
    set_up_logging(args.debug)

    if args.should_overwrite:
        logging.info("Overwriting test output files.")
    ret = main(args.project_dirs, args.should_overwrite, args.filter, coverage=None)
    sys.exit(ret)
