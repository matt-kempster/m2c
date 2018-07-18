import sys

import typing
from typing import List, Union, Iterator, Optional, Dict, Callable, Any

from flow_graph import *
from parse_file import parse_file
from translate import translate_to_ast
from if_statements import write_flowgraph


def main(filename: str, function_index: int) -> None:
    with open(filename, 'r') as f:
        mips_file: MIPSFile = parse_file(filename, f)
        flow_graph: FlowGraph = translate_to_ast(
            mips_file.functions[function_index])
        # Uncomment this to generate a graphviz rendering of the function.
        # visualize_callgraph(flow_graph)
        write_flowgraph(flow_graph)


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] in ['-h, --help']:
        print(f"USAGE: {sys.argv[0]} [filename] [function_index]")
    else:
        main(sys.argv[1], int(sys.argv[2]))
