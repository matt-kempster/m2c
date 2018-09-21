import sys
import argparse

from flow_graph import build_callgraph, visualize_callgraph
from parse_file import parse_file
from translate import translate_to_ast
from if_statements import write_function
from options import Options


def main(options: Options, function_index_or_name: str) -> None:
    with open(options.filename, 'r') as f:
        mips_file = parse_file(f, options)
        try:
            index = int(function_index_or_name)
            function = mips_file.functions[index]
        except ValueError:
            name = function_index_or_name
            try:
                function = next(f for f in mips_file.functions if f.name == name)
            except StopIteration:
                print(f"Function {name} not found.", file=sys.stderr)
                return
        except IndexError:
            count = len(mips_file.functions)
            print(f"Function index {index} is out of bounds (must be between " +
                    f"0 and {count - 1}).", file=sys.stderr)
            return

        # Uncomment this to generate a graphviz rendering of the function:
        # visualize_callgraph(build_callgraph(function))
        function_info = translate_to_ast(function, options)
        write_function(function_info, options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompile MIPS assembly to C.")
    parser.add_argument('filename', help="input filename")
    parser.add_argument('function', help="function index or name", type=str)
    parser.add_argument('--no-debug', dest='debug',
            help="don't print any debug info", action='store_false')
    parser.add_argument('--no-node-comments', dest='node_comments',
            help="don't print comments about node numbers", action='store_false')
    parser.add_argument('--stop-on-error', dest='stop_on_error',
            help="stop when encountering any error", action='store_true')
    args = parser.parse_args()
    options = Options(
        filename=args.filename,
        debug=args.debug,
        stop_on_error=args.stop_on_error,
        node_comments=args.node_comments,
    )
    main(options, args.function)
