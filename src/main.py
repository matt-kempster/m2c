import argparse
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union

from .c_types import TypeMap, build_typemap, dump_typemap
from .error import DecompFailure
from .flow_graph import visualize_flowgraph
from .if_statements import get_function_text
from .options import CodingStyle, Options
from .parse_file import AsmData, Function, parse_file
from .translate import (
    FunctionInfo,
    GlobalInfo,
    InstrProcessingFailure,
    translate_to_ast,
)


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
    all_functions: Dict[str, Function] = {}
    asm_data = AsmData()
    typemap: Optional[TypeMap] = None
    try:
        for filename in options.filenames:
            if filename == "-":
                mips_file = parse_file(sys.stdin, options)
            else:
                with open(filename, "r", encoding="utf-8-sig") as f:
                    mips_file = parse_file(f, options)
            all_functions.update((fn.name, fn) for fn in mips_file.functions)
            mips_file.asm_data.merge_into(asm_data)

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

    if not options.function_indexes_or_names:
        functions = list(all_functions.values())
    else:
        functions = []
        for index_or_name in options.function_indexes_or_names:
            if isinstance(index_or_name, int):
                if not (0 <= index_or_name < len(all_functions)):
                    print(
                        f"Function index {index_or_name} is out of bounds (must be between "
                        f"0 and {len(all_functions) - 1}).",
                        file=sys.stderr,
                    )
                    return 1
                functions.append(list(all_functions.values())[index_or_name])
            else:
                if index_or_name not in all_functions:
                    print(f"Function {index_or_name} not found.", file=sys.stderr)
                    return 1
                functions.append(all_functions[index_or_name])

    function_names = set(all_functions.keys())
    global_info = GlobalInfo(asm_data, function_names, typemap)
    function_infos: List[Union[FunctionInfo, Exception]] = []
    for function in functions:
        try:
            info = translate_to_ast(function, options, global_info)
            function_infos.append(info)
        except Exception as e:
            # Store the exception for later, to preserve the order in the output
            function_infos.append(e)

    if options.visualize_flowgraph:
        fn_info = function_infos[0]
        if isinstance(fn_info, Exception):
            raise fn_info
        print(visualize_flowgraph(fn_info.flow_graph))
        return 0

    fmt = options.formatter()
    global_decls = global_info.global_decls(fmt)
    if options.emit_globals and global_decls:
        print(global_decls)

    return_code = 0
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
            print("/*")
            print(f"Failed to decompile function {function.name}:\n")
            print(e)
            print("*/")
            return_code = 1
        except Exception:
            print("/*")
            print(f"Internal error while decompiling function {function.name}:\n")
            print_exception(sanitize=options.sanitize_tracebacks)
            print("*/")
            return_code = 1

    return return_code


def parse_flags(flags: List[str]) -> Options:
    parser = argparse.ArgumentParser(
        description="Decompile MIPS assembly to C.",
        usage="%(prog)s [--context C_FILE] [-f FN ...] filename [filename ...]",
    )

    group = parser.add_argument_group("Input Options")
    group.add_argument(
        "filename",
        nargs="+",
        help="input asm filename(s)",
    )
    group.add_argument(
        "--rodata",
        dest="filename",
        action="append",
        help=argparse.SUPPRESS,  # For backwards compatibility
    )
    group.add_argument(
        "--context",
        metavar="C_FILE",
        dest="c_context",
        help="read variable types/function signatures/structs from an existing C file. "
        "The file must already have been processed by the C preprocessor.",
    )
    group.add_argument(
        "-D",
        dest="defined",
        action="append",
        default=[],
        help="mark preprocessor constant as defined",
    )
    group.add_argument(
        "-U",
        dest="undefined",
        action="append",
        default=[],
        help="mark preprocessor constant as undefined",
    )

    group = parser.add_argument_group("Output & Formatting Options")
    group.add_argument(
        "-f",
        "--function",
        metavar="FN",
        dest="functions",
        action="append",
        default=[],
        help="function index or name to decompile",
    )
    group.add_argument(
        "--valid-syntax",
        dest="valid_syntax",
        action="store_true",
        help="emit valid C syntax, using macros to indicate unknown types or other "
        "unusual statements. Macro definitions are in `mips2c_macros.h`.",
    )
    group.add_argument(
        "--no-emit-globals",
        dest="emit_globals",
        action="store_false",
        help="do not emit global declarations with inferred types.",
    )
    group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="print debug info inline",
    )
    group.add_argument(
        "--print-assembly",
        dest="print_assembly",
        action="store_true",
        help="print assembly of function to decompile",
    )
    group.add_argument(
        "--allman",
        dest="allman",
        action="store_true",
        help="put braces on separate lines",
    )
    group.add_argument(
        "--pointer-style",
        dest="pointer_style",
        help="control whether to output pointer asterisks next to the type name (left) "
        "or next to the variable name (right)",
        choices=["left", "right"],
        default="right",
    )
    group.add_argument(
        "--unk-underscore",
        dest="unknown_underscore",
        help="emit unk_X instead of unkX for unknown struct accesses",
        action="store_true",
    )
    group.add_argument(
        "--dump-typemap",
        dest="dump_typemap",
        action="store_true",
        help="dump information about all functions and structs from the provided C "
        "context. Mainly useful for debugging.",
    )
    group.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="print an SVG visualization of the control flow graph using graphviz",
    )
    group.add_argument(
        "--sanitize-tracebacks",
        dest="sanitize_tracebacks",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    group = parser.add_argument_group("Analysis Options")
    group.add_argument(
        "--stop-on-error",
        dest="stop_on_error",
        action="store_true",
        help="stop when encountering any error",
    )
    group.add_argument(
        "--void",
        dest="void",
        action="store_true",
        help="assume the decompiled function returns void",
    )
    group.add_argument(
        "--gotos-only",
        dest="ifs",
        action="store_false",
        help="disable control flow generation; emit gotos for everything",
    )
    group.add_argument(
        "--no-ifs",
        dest="ifs",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--no-andor",
        dest="andor_detection",
        action="store_false",
        help="disable detection of &&/||",
    )
    group.add_argument(
        "--no-casts",
        dest="skip_casts",
        action="store_true",
        help="don't emit any type casts",
    )
    group.add_argument(
        "--reg-vars",
        metavar="REGISTERS",
        dest="reg_vars",
        help="use single variables instead of temps/phis for the given "
        "registers (comma separated)",
    )
    group.add_argument(
        "--goto",
        metavar="PATTERN",
        dest="goto_patterns",
        action="append",
        default=["GOTO"],
        help="emit gotos for branches on lines containing this substring "
        '(possibly within a comment). Default: "GOTO". Multiple '
        "patterns are allowed.",
    )
    group.add_argument(
        "--pdb-translate",
        dest="pdb_translate",
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
        pointer_style_left=args.pointer_style == "left",
        unknown_underscore=args.unknown_underscore,
    )
    filenames = args.filename

    # Backwards compatibility: giving a function index/name as a final argument, or "all"
    assert filenames, "checked by argparse, nargs='+'"
    if filenames[-1] == "all":
        filenames.pop()
    elif re.match(r"^[0-9a-zA-Z_]+$", filenames[-1]):
        # The filename is a valid C identifier or a number
        args.functions.append(filenames.pop())
    if not filenames:
        parser.error("the following arguments are required: filename")

    functions: List[Union[int, str]] = []
    for fn in args.functions:
        try:
            functions.append(int(fn))
        except ValueError:
            functions.append(fn)

    # The debug output interferes with the visualize output
    if args.visualize:
        args.debug = False

    return Options(
        filenames=filenames,
        function_indexes_or_names=functions,
        debug=args.debug,
        void=args.void,
        ifs=args.ifs,
        andor_detection=args.andor_detection,
        skip_casts=args.skip_casts,
        reg_vars=reg_vars,
        goto_patterns=args.goto_patterns,
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
