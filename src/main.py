import argparse
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union

from .error import DecompFailure
from .flow_graph import build_flowgraph, visualize_flowgraph
from .if_statements import get_function_text
from .options import Options, CodingStyle
from .parse_file import Function, MIPSFile, parse_file
from .translate import (
    FunctionInfo,
    GlobalInfo,
    InstrProcessingFailure,
    translate_to_ast,
)
from .types import Type
from .c_types import TypeMap, build_typemap, dump_typemap


def print_exception(sanitize: bool) -> None:
    """Print a traceback for the current exception to stdout.

    If `sanitize` is true, the filename's full path is stripped,
    and the line is set to 0. These changes make the test output
    less brittle."""
    if sanitize:
        tb = traceback.TracebackException(*sys.exc_info())
        if tb.exc_type == InstrProcessingFailure and tb.__cause__:
            tb = tb.__cause__
        for frame in tb.stack:
            frame.lineno = 0
            frame.filename = Path(frame.filename).name
        for line in tb.format(chain=False):
            print(line, end="")
    else:
        traceback.print_exc(file=sys.stdout)


def run(options: Options) -> int:
    mips_file: MIPSFile
    typemap: Optional[TypeMap] = None
    try:
        if options.filename == "-":
            mips_file = parse_file(sys.stdin, options)
        else:
            with open(options.filename, "r", encoding="utf-8-sig") as f:
                mips_file = parse_file(f, options)

        # Move over jtbl rodata from files given by --rodata & --data
        for data_file in options.data_files:
            with open(data_file, "r", encoding="utf-8-sig") as f:
                sub_file = parse_file(f, options)
                sub_file.data_section.merge_into(mips_file.data_section)

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
        functions = mips_file.functions
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
            functions = [mips_file.functions[index]]
        except ValueError:
            name = options.function_index_or_name
            functions = [fn for fn in mips_file.functions if fn.name == name]
            if not functions:
                print(f"Function {name} not found.", file=sys.stderr)
                return 1

    return_code = 0
    global_info = GlobalInfo(mips_file.data_section, typemap)
    function_infos: List[Union[FunctionInfo, Exception]] = []
    for function in functions:
        try:
            if options.visualize_flowgraph:
                visualize_flowgraph(build_flowgraph(function, mips_file.data_section))
                continue

            info = translate_to_ast(function, options, global_info)
            function_infos.append(info)
        except Exception as e:
            # Store the exception for later, to preserve the order in the output
            function_infos.append(e)

    if options.visualize_flowgraph:
        return return_code

    fmt = options.formatter()
    if options.emit_globals:
        global_decls = global_info.global_decls(fmt)
        if global_decls:
            print(global_decls)

    for index, (function, function_info) in enumerate(zip(functions, function_infos)):
        if index != 0:
            print()
        try:
            if options.print_assembly:
                print(function)
                print()

            if isinstance(function_info, Exception):
                raise function_info

            function_text = get_function_text(function_info, options)
            print(function_text)
        except DecompFailure as e:
            print(f"Failed to decompile function {function.name}:\n\n{e}")
            return_code = 1
        except Exception:
            print(f"Internal error while decompiling function {function.name}:\n")
            print_exception(sanitize=options.sanitize_tracebacks)
            return_code = 1

    return return_code


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
        "--gotos-only",
        dest="ifs",
        help="disable control flow generation; emit gotos for everything",
        action="store_false",
    )
    parser.add_argument(
        "--no-ifs",
        dest="ifs",
        action="store_false",
        help=argparse.SUPPRESS,
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
        # Deprecated, provided for backwards compatability. Use --data instead
        "--rodata",
        metavar="ASM_FILE",
        dest="data_files",
        action="append",
        default=[],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--data",
        metavar="ASM_FILE",
        dest="data_files",
        action="append",
        default=[],
        help="read jump table, constant, and global variable information from this file",
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
        "--emit-globals",
        dest="emit_globals",
        action="store_true",
        help="emit global declarations with inferred types.",
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
        data_files=args.data_files,
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
        emit_globals=args.emit_globals,
    )


def main() -> None:
    # Large functions can sometimes require a higher recursion limit than the
    # CPython default. Cap to INT_MAX to avoid an OverflowError, though.
    sys.setrecursionlimit(min(2 ** 31 - 1, 10 * sys.getrecursionlimit()))
    options = parse_flags(sys.argv[1:])
    sys.exit(run(options))


if __name__ == "__main__":
    main()
