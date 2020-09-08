#!/usr/bin/env python3
from coverage import Coverage  # type: ignore
import sys
import argparse
import os

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
args = parser.parse_args()

cov = Coverage(
    include="src/*", data_file=".coverage" if args.emit_data_file else None, branch=True
)
cov.start()

import run_tests

run_tests.set_up_logging(debug=False)
ret = run_tests.main(should_overwrite=False, coverage=cov)

cov.stop()

cov.html_report(directory=args.dir, show_contexts=True, skip_empty=True)
print(f"Wrote html to {args.dir}")

sys.exit(ret)
