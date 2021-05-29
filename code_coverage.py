#!/usr/bin/env python3
from coverage import Coverage  # type: ignore
import sys
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="Compute code coverage for tests.")
parser.add_argument(
    "--dir",
    dest="dir",
    help="output HTML to directory",
    default="htmlcov/",
)
parser.add_argument(
    "--emit-data-file",
    dest="emit_data_file",
    help="emit a .coverage file",
    action="store_true",
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

cov = Coverage(
    include="src/*", data_file=".coverage" if args.emit_data_file else None, branch=True
)
cov.start()

import run_tests

run_tests.set_up_logging(debug=False)
ret = run_tests.main(
    args.project_dirs, should_overwrite=False, filter_regex=args.filter, coverage=cov
)

cov.stop()

cov.html_report(directory=args.dir, show_contexts=True, skip_empty=True)
print(f"Wrote html to {args.dir}")

sys.exit(ret)
