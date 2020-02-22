import argparse
import sys

from .error import DecompFailure
from .flow_graph import build_flowgraph, visualize_flowgraph
from .if_statements import write_function
from .options import Options
from .parse_file import Function, Rodata, parse_file
from .translate import translate_to_ast


def decompile_function(options: Options, function: Function, rodata: Rodata) -> None:
    if options.print_assembly:
        print(function)
        print()

    if options.visualize_flowgraph:
        visualize_flowgraph(build_flowgraph(function, rodata))
        return

    function_info = translate_to_ast(function, options, rodata)
    write_function(function_info, options)


def run(options: Options, function_index_or_name: str) -> int:
    with open(options.filename, "r") as f:
        try:
            mips_file = parse_file(f, options)

            # Move over jtbl rodata from files given by --rodata
            for rodata_file in options.rodata_files:
                with open(rodata_file, "r") as f2:
                    sub_file = parse_file(f2, options)
                    for (sym, value) in sub_file.rodata.values.items():
                        mips_file.rodata.values[sym] = value
        except DecompFailure as e:
            print(e)
            return 1

        if function_index_or_name == "all":
            options.stop_on_error = True
            for fn in mips_file.functions:
                try:
                    decompile_function(options, fn, mips_file.rodata)
                except Exception:
                    print(f"{fn.name}: ERROR")
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
                    return 1
            except IndexError:
                count = len(mips_file.functions)
                print(
                    f"Function index {index} is out of bounds (must be between "
                    f"0 and {count - 1}).",
                    file=sys.stderr,
                )
                return 1

            try:
                decompile_function(options, function, mips_file.rodata)
            except DecompFailure as e:
                print(f"Failed to decompile function {function.name}:\n\n{e}")
                return 1
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Decompile MIPS assembly to C.")
    parser.add_argument("filename", help="input filename")
    parser.add_argument("function", help="function index or name (or 'all')", type=str)
    parser.add_argument(
        "--debug", dest="debug", help="print debug info", action="store_true"
    )
    parser.add_argument(
        "--void",
        dest="void",
        help="assume the decompiled function returns void",
        action="store_true",
    )
    parser.add_argument(
        "--no-ifs",
        dest="ifs",
        help="disable control flow generation; emit gotos for everything",
        action="store_false",
    )
    parser.add_argument(
        "--no-andor",
        dest="andor_detection",
        help="disable detection of &&/||",
        action="store_false",
    )
    parser.add_argument(
        "--goto",
        metavar="PATTERN",
        dest="goto_patterns",
        action="append",
        default=["GOTO"],
        help="emit gotos for branches on lines containing this substring "
        '(possibly within a comment). Default: "GOTO". Multiple '
        "patterns are allowed.",
    )
    parser.add_argument(
        "--rodata",
        metavar="ASM_FILE",
        dest="rodata_files",
        action="append",
        default=[],
        help="read jump table data from this file",
    )
    parser.add_argument(
        "--stop-on-error",
        dest="stop_on_error",
        help="stop when encountering any error",
        action="store_true",
    )
    parser.add_argument(
        "--print-assembly",
        dest="print_assembly",
        help="print assembly of function to decompile",
        action="store_true",
    )
    parser.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="display a visualization of the control flow graph using graphviz",
    )
    parser.add_argument(
        "-D",
        dest="defined",
        action="append",
        default=[],
        help="mark preprocessor constant as defined",
    )
    parser.add_argument(
        "-U",
        dest="undefined",
        action="append",
        default=[],
        help="mark preprocessor constant as undefined",
    )
    args = parser.parse_args()
    preproc_defines = {
        **{d: 0 for d in args.undefined},
        **{d.split("=")[0]: 1 for d in args.defined},
    }
    options = Options(
        filename=args.filename,
        debug=args.debug,
        void=args.void,
        ifs=args.ifs,
        andor_detection=args.andor_detection,
        goto_patterns=args.goto_patterns,
        rodata_files=args.rodata_files,
        stop_on_error=args.stop_on_error,
        print_assembly=args.print_assembly,
        visualize_flowgraph=args.visualize,
        preproc_defines=preproc_defines,
    )
    return run(options, args.function)


if __name__ == "__main__":
    sys.exit(main())
