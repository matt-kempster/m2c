import argparse
import sys

from error import DecompFailure
from flow_graph import build_flowgraph, visualize_flowgraph
from if_statements import write_function
from options import Options
from parse_file import Function, parse_file
from translate import translate_to_ast


def decompile_function(options: Options, function: Function) -> None:
    if options.print_assembly:
        print(function)
        print()

    if options.visualize_flowgraph:
        visualize_flowgraph(build_flowgraph(function))
        return

    function_info = translate_to_ast(function, options)
    write_function(function_info, options)


def main(options: Options, function_index_or_name: str) -> None:
    with open(options.filename, 'r') as f:
        mips_file = parse_file(f, options)
        if function_index_or_name == 'all':
            options.stop_on_error = True
            for fn in mips_file.functions:
                try:
                    decompile_function(options, fn)
                except Exception:
                    print(f'{fn.name}: ERROR')
                print()
        else:
            try:
                index = int(function_index_or_name)
                function = mips_file.functions[index]
            except ValueError:
                name = function_index_or_name
                try:
                    function = next(f for f in mips_file.functions if f.name == name)
                except StopIteration:
                    print(f"Function {name} not found.", file=sys.stderr)
                    exit(1)
            except IndexError:
                count = len(mips_file.functions)
                print(f"Function index {index} is out of bounds (must be between "
                        f"0 and {count - 1}).", file=sys.stderr)
                exit(1)

            try:
                decompile_function(options, function)
            except DecompFailure as e:
                print(f"Failed to decompile function {function.name}:\n\n{e}")
                exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompile MIPS assembly to C.")
    parser.add_argument('filename', help="input filename")
    parser.add_argument('function', help="function index or name (or 'all')", type=str)
    parser.add_argument('--debug', dest='debug',
            help="print debug info", action='store_true')
    parser.add_argument('--no-ifs', dest='ifs',
            help="disable control flow generation; emit gotos for everything", action='store_false')
    parser.add_argument('--no-andor', dest='andor_detection',
            help="disable detection of &&/||", action='store_false')
    parser.add_argument('--stop-on-error', dest='stop_on_error',
            help="stop when encountering any error", action='store_true')
    parser.add_argument('--print-assembly', dest='print_assembly',
            help="print assembly of function to decompile", action='store_true')
    parser.add_argument('--visualize', dest='visualize', action='store_true',
            help="display a visualization of the control flow graph using graphviz")
    parser.add_argument('-D', dest='defined', action='append',
            help="mark preprocessor constant as defined")
    parser.add_argument('-U', dest='undefined', action='append',
            help="mark preprocessor constant as undefined")
    args = parser.parse_args()
    preproc_defines = {
        **{d: 0 for d in (args.undefined or [])},
        **{d.split('=')[0]: 1 for d in (args.defined or [])},
    }
    options = Options(
        filename=args.filename,
        debug=args.debug,
        andor_detection=args.andor_detection,
        ifs=args.ifs,
        stop_on_error=args.stop_on_error,
        print_assembly=args.print_assembly,
        visualize_flowgraph=args.visualize,
        preproc_defines=preproc_defines,
    )
    main(options, args.function)
