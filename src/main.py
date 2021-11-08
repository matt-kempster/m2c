import argparse
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union

from .c_types import TypeMap, build_typemap, dump_typemap
from .error import DecompFailure
from .flow_graph import FlowGraph, build_flowgraph, visualize_flowgraph
from .if_statements import get_function_text
from .options import CodingStyle, Options
from .parse_file import AsmData, Function, parse_file
from .translate import (
    FunctionInfo,
    GlobalInfo,
    InstrProcessingFailure,
    translate_to_ast,
)
from .types import TypePool


def print_current_exception(sanitize: bool) -> None:
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


def print_exception_as_comment(
    exc: Exception, context: Optional[str], sanitize: bool
) -> None:
    context_phrase = f" in {context}" if context is not None else ""
    if isinstance(exc, OSError):
        print(f"/* OSError{context_phrase}: {exc} */")
        return
    elif isinstance(exc, DecompFailure):
        print("/*")
        print(f"Decompilation failure{context_phrase}:\n")
        print(exc)
        print("*/")
    else:
        print("/*")
        print(f"Internal error{context_phrase}:\n")
        print_current_exception(sanitize=sanitize)
        print("*/")


def run(options: Options) -> int:
    all_functions: Dict[str, Function] = {}
    asm_data = AsmData()
    try:
        for filename in options.filenames:
            if filename == "-":
                mips_file = parse_file(sys.stdin, options)
            else:
                with open(filename, "r", encoding="utf-8-sig") as f:
                    mips_file = parse_file(f, options)
            all_functions.update((fn.name, fn) for fn in mips_file.functions)
            mips_file.asm_data.merge_into(asm_data)

        typemap = build_typemap(options.c_contexts, use_cache=options.use_cache)
    except Exception as e:
        print_exception_as_comment(
            e, context=None, sanitize=options.sanitize_tracebacks
        )
        return 1

    if options.dump_typemap:
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

    fmt = options.formatter()
    function_names = set(all_functions.keys())
    typepool = TypePool(
        unknown_field_prefix="unk_" if fmt.coding_style.unknown_underscore else "unk",
        struct_field_inference=options.struct_field_inference,
    )
    global_info = GlobalInfo(asm_data, function_names, typemap, typepool)

    flow_graphs: List[Union[FlowGraph, Exception]] = []
    for function in functions:
        try:
            flow_graphs.append(build_flowgraph(function, global_info.asm_data))
        except Exception as e:
            # Store the exception for later, to preserve the order in the output
            flow_graphs.append(e)

    # Perform the preliminary passes to improve type resolution, but discard the results/exceptions
    for i in range(options.passes - 1):
        preliminary_infos = []
        for function, flow_graph in zip(functions, flow_graphs):
            try:
                if isinstance(flow_graph, Exception):
                    raise flow_graph
                flow_graph.reset_block_info()
                info = translate_to_ast(function, flow_graph, options, global_info)
                preliminary_infos.append(info)
            except:
                pass
        try:
            global_info.global_decls(fmt, options.global_decls)
        except:
            pass
        for info in preliminary_infos:
            try:
                get_function_text(info, options)
            except:
                pass

    function_infos: List[Union[FunctionInfo, Exception]] = []
    for function, flow_graph in zip(functions, flow_graphs):
        try:
            if isinstance(flow_graph, Exception):
                raise flow_graph
            flow_graph.reset_block_info()
            info = translate_to_ast(function, flow_graph, options, global_info)
            function_infos.append(info)
        except Exception as e:
            # Store the exception for later, to preserve the order in the output
            function_infos.append(e)

    return_code = 0
    try:
        if options.visualize_flowgraph:
            fn_info = function_infos[0]
            if isinstance(fn_info, Exception):
                raise fn_info
            print(visualize_flowgraph(fn_info.flow_graph))
            return 0

        if options.structs:
            type_decls = typepool.format_type_declarations(fmt)
            if type_decls:
                print(type_decls)

        global_decls = global_info.global_decls(fmt, options.global_decls)
        if global_decls:
            print(global_decls)
    except Exception as e:
        print_exception_as_comment(
            e, context=None, sanitize=options.sanitize_tracebacks
        )
        return_code = 1

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
        except Exception as e:
            print_exception_as_comment(
                e,
                context=f"function {function.name}",
                sanitize=options.sanitize_tracebacks,
            )
            return_code = 1

    for warning in typepool.warnings:
        print(fmt.with_comments("", comments=[warning]))

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
        help="Input asm filename(s)",
    )
    group.add_argument(
        "--rodata",
        dest="rodata_filenames",
        action="append",
        default=[],
        help=argparse.SUPPRESS,  # For backwards compatibility
    )
    group.add_argument(
        "--context",
        metavar="C_FILE",
        dest="c_contexts",
        action="append",
        type=Path,
        default=[],
        help="Read variable types/function signatures/structs from an existing C file. "
        "The file must already have been processed by the C preprocessor.",
    )
    group.add_argument(
        "--no-cache",
        action="store_false",
        dest="use_cache",
        help="Disable caching of variable types/function signatures/structs from the parsed C context. "
        "This option should be used for untrusted environments. "
        'The cache for "foo/ctx_bar.c" is stored in "foo/ctx_bar.c.m2c". '
        "The *.m2c files automatically regenerate when the source file change, and can be ignored.",
    )
    group.add_argument(
        "-D",
        dest="defined",
        action="append",
        default=[],
        help="Mark preprocessor constant as defined",
    )
    group.add_argument(
        "-U",
        dest="undefined",
        action="append",
        default=[],
        help="Mark preprocessor constant as undefined",
    )

    group = parser.add_argument_group("Output Options")
    group.add_argument(
        "-f",
        "--function",
        metavar="FN",
        dest="functions",
        action="append",
        default=[],
        help="Function index or name to decompile",
    )
    group.add_argument(
        "--globals",
        dest="global_decls",
        type=Options.GlobalDeclsEnum,
        choices=list(Options.GlobalDeclsEnum),
        default="used",
        help="Control which global declarations & initializers are emitted. "
        '"all" includes all globals with entries in .data/.rodata/.bss, as well as inferred symbols. '
        '"used" only includes symbols used by the decompiled functions that are not in the context (default). '
        '"none" does not emit any global declarations. ',
    )
    group.add_argument(
        "--structs",
        dest="structs",
        action="store_true",
        help="Perform type inference on unknown struct fields, and include struct declarations "
        "representing each function's stack in the output. These can be modified and passed back "
        "to mips_to_c via --context to improve the output.",
    )
    group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        help="Print debug info inline",
    )
    group.add_argument(
        "--print-assembly",
        dest="print_assembly",
        action="store_true",
        help="Print assembly of function to decompile",
    )
    group.add_argument(
        "--dump-typemap",
        dest="dump_typemap",
        action="store_true",
        help="Dump information about all functions and structs from the provided C "
        "context. Mainly useful for debugging.",
    )
    group.add_argument(
        "--visualize",
        dest="visualize",
        action="store_true",
        help="Print an SVG visualization of the control flow graph using graphviz",
    )
    group.add_argument(
        "--sanitize-tracebacks",
        dest="sanitize_tracebacks",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    group = parser.add_argument_group("Formatting Options")
    group.add_argument(
        "--valid-syntax",
        dest="valid_syntax",
        action="store_true",
        help="Emit valid C syntax, using macros to indicate unknown types or other "
        "unusual statements. Macro definitions are in `mips2c_macros.h`.",
    )
    group.add_argument(
        "--allman",
        dest="allman",
        action="store_true",
        help="Put braces on separate lines",
    )
    group.add_argument(
        "--pointer-style",
        dest="pointer_style",
        help="Control whether to output pointer asterisks next to the type name (left) "
        "or next to the variable name (right). Default: right",
        choices=["left", "right"],
        default="right",
    )
    group.add_argument(
        "--unk-underscore",
        dest="unknown_underscore",
        help="Emit unk_X instead of unkX for unknown struct accesses",
        action="store_true",
    )
    group.add_argument(
        "--hex-case",
        dest="hex_case",
        help="Display case labels in hex rather than decimal",
        action="store_true",
    )
    group.add_argument(
        "--comment-style",
        dest="comment_style",
        choices=["multiline", "oneline"],
        default="multiline",
        help='Comment formatting. "multiline" for C-style `/* ... */`, "oneline" for C++-style `// ...`. '
        "Default: multiline",
    )
    group.add_argument(
        "--comment-column",
        dest="comment_column",
        metavar="N",
        type=int,
        default=52,
        help="Column number to justify comments to. Set to 0 to disable justification. Default: 52",
    )
    group.add_argument(
        "--no-casts",
        dest="skip_casts",
        action="store_true",
        help="Don't emit any type casts",
    )

    group = parser.add_argument_group("Analysis Options")
    group.add_argument(
        "--passes",
        "-P",
        dest="passes",
        metavar="N",
        type=int,
        default=2,
        help="Number of translation passes to perform. Each pass may improve type resolution and produce better "
        "output, particularly when decompiling multiple functions. Default: 2",
    )
    group.add_argument(
        "--compiler",
        dest="compiler",
        type=Options.CompilerEnum,
        choices=list(Options.CompilerEnum),
        default="ido",
        help="Original compiler family that produced the input files. "
        "Used when the compiler's behavior cannot be inferred from the input, e.g. stack ordering. "
        "Default: ido",
    )
    group.add_argument(
        "--stop-on-error",
        dest="stop_on_error",
        action="store_true",
        help="Stop when encountering any error",
    )
    group.add_argument(
        "--void",
        dest="void",
        action="store_true",
        help="Assume the decompiled function returns void",
    )
    group.add_argument(
        "--gotos-only",
        dest="ifs",
        action="store_false",
        help="Disable control flow generation; emit gotos for everything",
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
        help="Disable detection of &&/||",
    )
    group.add_argument(
        "--no-struct-inference",
        dest="no_struct_inference",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    group.add_argument(
        "--reg-vars",
        metavar="REGISTERS",
        dest="reg_vars",
        help="Use single variables instead of temps/phis for the given "
        "registers (comma separated)",
    )
    group.add_argument(
        "--goto",
        metavar="PATTERN",
        dest="goto_patterns",
        action="append",
        default=["GOTO"],
        help="Emit gotos for branches on lines containing this substring "
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
        hex_case=args.hex_case,
        oneline_comments=args.comment_style == "oneline",
        comment_column=args.comment_column,
    )
    filenames = args.filename + args.rodata_filenames

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
        c_contexts=args.c_contexts,
        use_cache=args.use_cache,
        dump_typemap=args.dump_typemap,
        pdb_translate=args.pdb_translate,
        preproc_defines=preproc_defines,
        coding_style=coding_style,
        sanitize_tracebacks=args.sanitize_tracebacks,
        valid_syntax=args.valid_syntax,
        global_decls=args.global_decls,
        compiler=args.compiler,
        structs=args.structs,
        struct_field_inference=args.structs and not args.no_struct_inference,
        passes=args.passes,
    )


def main() -> None:
    # Large functions can sometimes require a higher recursion limit than the
    # CPython default. Cap to INT_MAX to avoid an OverflowError, though.
    sys.setrecursionlimit(min(2 ** 31 - 1, 10 * sys.getrecursionlimit()))
    options = parse_flags(sys.argv[1:])
    sys.exit(run(options))


if __name__ == "__main__":
    main()
