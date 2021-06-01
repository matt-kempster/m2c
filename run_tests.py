#!/usr/bin/env python3
import argparse
import attr
import contextlib
import difflib
import io
import logging
import re
import shlex
import sys
from coverage import Coverage  # type: ignore
from pathlib import Path
from typing import Any, List, Optional, Pattern, Tuple

from src.options import Options

CRASH_STRING = "CRASHED\n"


@attr.s
class TestOptions:
    should_overwrite: bool = attr.ib()
    diff_context: int = attr.ib()
    filter_re: Pattern[str] = attr.ib()
    coverage: Any = attr.ib()


@attr.s
class TestCase:
    name: str = attr.ib()
    asm_file: Path = attr.ib()
    output_file: Path = attr.ib()
    brief_crashes: bool = attr.ib(default=True)
    flags_path: Optional[Path] = attr.ib(default=None)
    flags: List[str] = attr.ib(factory=list)


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


def decompile_and_compare(test_case: TestCase, test_options: TestOptions) -> bool:
    # This import is deferred so it can be profiled by the coverage tool
    from src.main import parse_flags

    logging.info(f"Running test: {test_case.name}")
    logging.debug(
        f"Decompiling {test_case.asm_file}"
        + (f" into {test_case.output_file}" if test_options.should_overwrite else "")
    )
    try:
        original_contents = test_case.output_file.read_text()
    except FileNotFoundError:
        if not test_options.should_overwrite:
            logging.error(f"{test_case.output_file} does not exist. Skipping.")
            return True
        logging.info(f"{test_case.output_file} does not exist. Creating...")
        original_contents = "(file did not exist)"

    test_flags = ["--sanitize-tracebacks", "--stop-on-error", str(test_case.asm_file)]
    test_flags.extend(test_case.flags)
    if test_case.flags_path is not None:
        test_flags.extend(get_test_flags(test_case.flags_path))
    options = parse_flags(test_flags)

    final_contents = decompile_and_capture_output(options, test_case.brief_crashes)

    if test_options.should_overwrite:
        test_case.output_file.parent.mkdir(parents=True, exist_ok=True)
        test_case.output_file.write_text(final_contents)

    changed = final_contents != original_contents
    if changed:
        logging.info(
            "\n".join(
                [
                    f"Output of {test_case.asm_file} changed! Diff:",
                    *difflib.unified_diff(
                        original_contents.splitlines(),
                        final_contents.splitlines(),
                        n=test_options.diff_context,
                    ),
                ]
            )
        )
    return not changed


def decompile_and_capture_output(options: Options, brief_crashes: bool) -> str:
    # This import is deferred so it can be profiled by the coverage tool
    from src.main import run as decompile

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


def create_e2e_tests(
    e2e_top_dir: Path,
    e2e_test_path: Path,
) -> List[TestCase]:

    cases: List[TestCase] = []
    for asm_file in e2e_test_path.glob("*.s"):
        output_file = asm_file.parent.joinpath(asm_file.stem + "-out.c")
        flags_path = asm_file.parent.joinpath(asm_file.stem + "-flags.txt")
        name = f"e2e:{asm_file.relative_to(e2e_top_dir)}"

        cases.append(
            TestCase(
                name=name,
                asm_file=asm_file,
                output_file=output_file,
                brief_crashes=True,
                flags_path=flags_path,
                flags=["test"],  # Decompile the function 'test'
            )
        )
    return cases


def create_project_tests(
    base_dir: Path,
    output_dir: Path,
    context_file: Optional[Path],
    name_prefix: str,
) -> List[TestCase]:
    cases: List[TestCase] = []
    asm_dir = base_dir / "asm"
    for asm_file in asm_dir.rglob("*"):
        if asm_file.suffix not in (".asm", ".s"):
            continue

        asm_name = asm_file.name
        if (
            asm_name.startswith("code_data")
            or asm_name.startswith("code_rodata")
            or asm_name.startswith("boot_data")
            or asm_name.startswith("boot_rodata")
            or asm_name.endswith("_data.asm")
            or asm_name.endswith("_rodata.asm")
            or asm_name.endswith(".data.s")
            or asm_name.endswith(".rodata.s")
            or asm_name.endswith(".rodata2.s")
        ):
            continue

        flags = []
        if context_file is not None:
            flags.extend(["--context", str(context_file)])

        # Guess the name of .rodata file(s) for the MM decomp project
        for candidate in [
            # mm code/*.asm
            "code_rodata_" + asm_name,
            asm_name.replace("code_", "code_rodata_"),
            # mm boot/*.asm
            "boot_rodata_" + asm_name,
            asm_name.replace("boot_", "boot_rodata_"),
            # mm overlays/*.asm
            asm_name.rpartition("_0x")[0] + "_rodata.asm",
            asm_name.rpartition("_0x")[0] + "_late_rodata.asm",
            # oot *.s
            asm_name.replace(".s", ".rodata.s"),
            asm_name.replace(".s", ".rodata2.s"),
        ]:
            if candidate == asm_name:
                continue
            f = asm_file.parent / candidate
            if f.exists():
                flags.extend(["--rodata", str(f)])

        test_path = asm_file.relative_to(asm_dir)
        name = f"{name_prefix}:{test_path}"
        output_file = (output_dir / test_path).with_suffix(".c")

        cases.append(
            TestCase(
                name=name,
                asm_file=asm_file,
                output_file=output_file,
                brief_crashes=False,
                flags=flags,
            )
        )
    return cases


def main(
    project_dirs: List[Tuple[Path, bool]],
    options: TestOptions,
) -> int:
    # Collect tests
    test_cases: List[TestCase] = []

    e2e_top_dir = Path(__file__).parent / "tests" / "end_to_end"
    for e2e_test_path in e2e_top_dir.iterdir():
        test_cases.extend(create_e2e_tests(e2e_top_dir, e2e_test_path))

    for project_dir, use_context in project_dirs:
        name_prefix = project_dir.name
        if project_dir.match("papermario/ver/us"):
            name_prefix = "papermario_us"
        elif project_dir.match("papermario/ver/jp"):
            name_prefix = "papermario_jp"

        context_file: Optional[Path] = None
        if use_context:
            name_prefix = f"{name_prefix}_ctx"
            context_file = project_dir / "ctx.c"
            if not context_file.exists():
                raise Exception(
                    f"{project_dir} tests require context file, but {context_file} does not exist"
                )

        output_dir = Path(__file__).parent / "tests" / "project" / name_prefix

        test_cases.extend(
            create_project_tests(
                project_dir,
                output_dir,
                context_file,
                name_prefix,
            )
        )

    ret = 0
    passed, skipped, failed = 0, 0, 0
    for test_case in test_cases:
        if options.filter_re is not None:
            if not options.filter_re.search(test_case.name):
                skipped += 1
                continue

        if decompile_and_compare(test_case, options):
            passed += 1
        else:
            failed += 1
            if options.should_overwrite:
                ret = 1

    logging.info(
        f"Test summary: {passed} passed, {skipped} skipped, {failed} failed, {passed + skipped + failed} total"
    )
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run and record end-to-end decompiler tests."
    )
    parser.add_argument(
        "--debug", dest="debug", help="print debug info", action="store_true"
    )
    parser.add_argument(
        "--diff-context",
        dest="diff_context",
        default=3,
        type=int,
        help=("Number of lines of context to print with in diff output."),
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
        dest="filter_re",
        type=lambda x: re.compile(x),
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
    cov_group = parser.add_argument_group("Coverage")
    cov_group.add_argument(
        "--coverage",
        dest="coverage",
        action="store_true",
        help="Compute code coverage for tests",
    )
    cov_group.add_argument(
        "--coverage-html",
        dest="coverage_html",
        help="Output coverage HTML report to directory",
        default="htmlcov/",
    )
    cov_group.add_argument(
        "--coverage-emit-data",
        dest="coverage_emit_data",
        action="store_true",
        help="Emit a .coverage data file",
    )
    args = parser.parse_args()
    set_up_logging(args.debug)

    cov = None
    if args.coverage:
        logging.info("Computing code coverage.")
        coverage_data_file = None
        if args.coverage_emit_data:
            coverage_data_file = ".coverage"
            logging.info(f"Writing coverage data to {coverage_data_file}")
        cov = Coverage(include="src/*", data_file=coverage_data_file, branch=True)
        cov.start()

    if args.should_overwrite:
        logging.info("Overwriting test output files.")

    options = TestOptions(
        should_overwrite=args.should_overwrite,
        diff_context=args.diff_context,
        filter_re=args.filter_re,
        coverage=cov,
    )
    ret = main(args.project_dirs, options)

    if cov is not None:
        cov.stop()
        cov.html_report(
            directory=args.coverage_html, show_contexts=True, skip_empty=True
        )
        logging.info(f"Wrote html to {args.coverage_html}")

    sys.exit(ret)
