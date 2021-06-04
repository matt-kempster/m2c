import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

from .error import DecompFailure
from .flow_graph import build_flowgraph, visualize_flowgraph
from .if_statements import get_function_text
from .options import Options, CodingStyle
from .parse_file import Function, MIPSFile, Rodata, parse_file
from .translate import translate_to_ast
from .c_types import TypeMap, build_typemap, dump_typemap


def print_exception(sanitize: bool) -> None:
    """Print a traceback for the current exception to stdout.

    If `sanitize` is true, the filename's full path is stripped,
    and the line is set to 0. These changes make the test output
    less brittle."""
    if sanitize:
        exc = sys.exc_info()
        frames = traceback.extract_tb(exc[2])
        for frame in frames:
            frame.lineno = 0
            frame.filename = Path(frame.filename).name

        print("Traceback (most recent call last):")
        print(
            "".join(
                traceback.format_list(frames)
                + traceback.format_exception_only(exc[0], exc[1])
            ),
            end="",
        )
    else:
        traceback.print_exc(file=sys.stdout)


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
        if options.filename == "-":
            mips_file = parse_file(sys.stdin, options)
        else:
            with open(options.filename, "r", encoding="utf-8-sig") as f:
                mips_file = parse_file(f, options)

        # Move over jtbl rodata from files given by --rodata
        for rodata_file in options.rodata_files:
            with open(rodata_file, "r", encoding="utf-8-sig") as f:
                sub_file = parse_file(f, options)
                sub_file.rodata.merge_into(mips_file.rodata)

        if options.c_context is not None:
            with open(options.c_context, "r", encoding="utf-8-sig") as f:
                typemap = build_typemap(f.read())
    except (OSError, DecompFailure) as e:
        print(e)
        return 1
    except Exception as e:
        print_exception(sanitize=options.sanitize_tracebacks)
        return 1

    if options.dump_typemap:
        assert typemap
        dump_typemap(typemap)
        return 0

    if options.function_index_or_name is None:
        has_error = False
        for index, fn in enumerate(mips_file.functions):
            if index != 0:
                print()
            try:
                decompile_function(options, fn, mips_file.rodata, typemap)
            except DecompFailure as e:
                print(f"Failed to decompile function {fn.name}:\n\n{e}")
                has_error = True
            except Exception:
                print(f"Internal error while decompiling function {fn.name}:\n")
                print_exception(sanitize=options.sanitize_tracebacks)
                has_error = True
        if has_error:
            return 1
    else:
        try:
            index = int(options.function_index_or_name)
            count = len(mips_file.functions)
            if not (0 <= index < count):
                print(
                    f"Function index {index} is out of bounds (must be between "
                    f"0 and {count - 1}).",
                    file=sys.stderr,
                )
                return 1
            function = mips_file.functions[index]
        except ValueError:
            name = options.function_index_or_name
            try:
                function = next(fn for fn in mips_file.functions if fn.name == name)
            except StopIteration:
                print(f"Function {name} not found.", file=sys.stderr)
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
    parser.add_argument("function", help="function index or name", nargs="?")
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
        "--no-casts",
        dest="skip_casts",
        help="don't emit any type casts",
        action="store_true",
    )
    parser.add_argument(
        "--reg-vars",
        metavar="REGISTERS",
        dest="reg_vars",
        help="use single variables instead of temps/phis for the given "
        "registers (comma separated)",
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
        help="dump information about all functions and structs from the provided C "
        "context. Mainly useful for debugging.",
    )
    parser.add_argument(
        "--valid-syntax",
        dest="valid_syntax",
        action="store_true",
        help="emit valid C syntax, using macros to indicate unknown types or other "
        "unusual statements. Macro definitions are in `mips2c_macros.h`.",
    )
    parser.add_argument(
        "--pdb-translate",
        dest="pdb_translate",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sanitize-tracebacks",
        dest="sanitize_tracebacks",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(flags)
    reg_vars = args.reg_vars.split(",") if args.reg_vars else []
    preproc_defines = {
        **{d: 0 for d in args.undefined},
        **{d.split("=")[0]: 1 for d in args.defined},
    }
    coding_style = CodingStyle(
        newline_after_function=args.allman,
        newline_after_if=args.allman,
        newline_before_else=args.allman,
    )
    function = args.function
    if function == "all":
        # accept "all" as "all functions in file", for compat reasons.
        function = None
    return Options(
        filename=args.filename,
        function_index_or_name=function,
        debug=args.debug,
        void=args.void,
        ifs=args.ifs,
        andor_detection=args.andor_detection,
        skip_casts=args.skip_casts,
        reg_vars=reg_vars,
        goto_patterns=args.goto_patterns,
        rodata_files=args.rodata_files,
        stop_on_error=args.stop_on_error,
        print_assembly=args.print_assembly,
        visualize_flowgraph=args.visualize,
        c_context=args.c_context,
        dump_typemap=args.dump_typemap,
        pdb_translate=args.pdb_translate,
        preproc_defines=preproc_defines,
        coding_style=coding_style,
        sanitize_tracebacks=args.sanitize_tracebacks,
        valid_syntax=args.valid_syntax,
    )


def main() -> None:
    # Large functions can sometimes require a higher recursion limit than the
    # CPython default. Cap to INT_MAX to avoid an OverflowError, though.
    sys.setrecursionlimit(min(2 ** 31 - 1, 10 * sys.getrecursionlimit()))
    options = parse_flags(sys.argv[1:])
    sys.exit(run(options))


if __name__ == "__main__":
    main()
