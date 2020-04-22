import argparse
import sys
from typing import List, Optional

from .error import DecompFailure
from .flow_graph import build_flowgraph, visualize_flowgraph
from .if_statements import get_function_text
from .options import Options, CodingStyle
from .parse_file import Function, MIPSFile, Rodata, parse_file
from .translate import translate_to_ast
from .c_types import TypeMap, build_typemap, dump_typemap


def decompile_function(
    options: Options, function: Function, rodata: Rodata, typemap: Optional[TypeMap]
) -> None:
    if options.print_assembly:
        print(function)
        print()

    if options.visualize_flowgraph:
        visualize_flowgraph(build_flowgraph(function, rodata))
        return

    function_info = translate_to_ast(function, options, rodata, typemap)
    function_text = get_function_text(function_info, options)
    print(function_text)


def run(options: Options) -> int:
    mips_file: MIPSFile
    typemap: Optional[TypeMap] = None
    try:
        with open(options.filename, "r", encoding="utf-8-sig") as f:
            mips_file = parse_file(f, options)

        # Move over jtbl rodata from files given by --rodata
        for rodata_file in options.rodata_files:
            with open(rodata_file, "r", encoding="utf-8-sig") as f:
                sub_file = parse_file(f, options)
                for (sym, value) in sub_file.rodata.values.items():
                    mips_file.rodata.values[sym] = value

        if options.c_context is not None:
            with open(options.c_context, "r", encoding="utf-8-sig") as f:
                typemap = build_typemap(f.read())
    except (OSError, DecompFailure) as e:
        print(e)
        return 1

    if options.dump_typemap:
        assert typemap
        dump_typemap(typemap)
        return 0

    if options.function_index_or_name == "all":
        options.stop_on_error = True
        for fn in mips_file.functions:
            try:
                decompile_function(options, fn, mips_file.rodata, typemap)
            except Exception:
                print(f"{fn.name}: ERROR")
            print()
    else:
        try:
            index = int(options.function_index_or_name)
            function = mips_file.functions[index]
        except ValueError:
            name = options.function_index_or_name
            try:
                function = next(fn for fn in mips_file.functions if fn.name == name)
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
            decompile_function(options, function, mips_file.rodata, typemap)
        except DecompFailure as e:
            print(f"Failed to decompile function {function.name}:\n\n{e}")
            return 1
    return 0


def parse_flags(flags: List[str]) -> Options:
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
        "--allman",
        dest="allman",
        action="store_true",
        help="put braces on separate lines",
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
    parser.add_argument(
        "--context",
        metavar="C_FILE",
        dest="c_context",
        help="read variable types/function signatures/structs from an existing C file. "
        "The file must already have been processed by the C preprocessor.",
    )
    parser.add_argument(
        "--dump-typemap",
        dest="dump_typemap",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(flags)
    preproc_defines = {
        **{d: 0 for d in args.undefined},
        **{d.split("=")[0]: 1 for d in args.defined},
    }
    coding_style = CodingStyle(
        newline_after_function=args.allman,
        newline_after_if=args.allman,
        newline_before_else=args.allman,
    )
    return Options(
        filename=args.filename,
        function_index_or_name=args.function,
        debug=args.debug,
        void=args.void,
        ifs=args.ifs,
        andor_detection=args.andor_detection,
        goto_patterns=args.goto_patterns,
        rodata_files=args.rodata_files,
        stop_on_error=args.stop_on_error,
        print_assembly=args.print_assembly,
        visualize_flowgraph=args.visualize,
        c_context=args.c_context,
        dump_typemap=args.dump_typemap,
        preproc_defines=preproc_defines,
        coding_style=coding_style,
    )


def main() -> int:
    options = parse_flags(sys.argv[1:])
    sys.exit(run(options))


if __name__ == "__main__":
    main()
