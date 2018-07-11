import sys

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any

from flow_graph import *
from parse_file import parse_file
from translate import translate_to_ast


def main(filename: str) -> None:
    with open(filename, 'r') as f:
        mips_file: MIPSFile = parse_file(filename, f)
        translate_to_ast(mips_file.functions[1])


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ['-h, --help']:
        print(f"USAGE: {sys.argv[0]} [filename]")
    else:
        main(sys.argv[1])
