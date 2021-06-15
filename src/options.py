import contextlib
from typing import Dict, Iterator, List, Optional

import attr


@attr.s
class CodingStyle:
    newline_after_function: bool = attr.ib()
    newline_after_if: bool = attr.ib()
    newline_before_else: bool = attr.ib()


@attr.s
class Options:
    filename: str = attr.ib()
    function_index_or_name: Optional[str] = attr.ib()
    debug: bool = attr.ib()
    void: bool = attr.ib()
    ifs: bool = attr.ib()
    andor_detection: bool = attr.ib()
    skip_casts: bool = attr.ib()
    reg_vars: List[str] = attr.ib()
    goto_patterns: List[str] = attr.ib()
    asm_data_files: List[str] = attr.ib()
    stop_on_error: bool = attr.ib()
    print_assembly: bool = attr.ib()
    visualize_flowgraph: bool = attr.ib()
    c_context: Optional[str] = attr.ib()
    dump_typemap: bool = attr.ib()
    pdb_translate: bool = attr.ib()
    preproc_defines: Dict[str, int] = attr.ib()
    coding_style: CodingStyle = attr.ib()
    sanitize_tracebacks: bool = attr.ib()
    valid_syntax: bool = attr.ib()
    emit_globals: bool = attr.ib()

    def formatter(self) -> "Formatter":
        return Formatter(
            self.coding_style,
            skip_casts=self.skip_casts,
            valid_syntax=self.valid_syntax,
        )


DEFAULT_CODING_STYLE: CodingStyle = CodingStyle(
    newline_after_function=False,
    newline_after_if=False,
    newline_before_else=False,
)


@attr.s
class Formatter:
    coding_style: CodingStyle = attr.ib(default=DEFAULT_CODING_STYLE)
    indent_step: str = attr.ib(default=" " * 4)
    skip_casts: bool = attr.ib(default=False)
    extra_indent: int = attr.ib(default=0)
    debug: bool = attr.ib(default=False)
    valid_syntax: bool = attr.ib(default=False)

    def indent(self, indent: int, line: str) -> str:
        return self.indent_step * max(indent + self.extra_indent, 0) + line

    @contextlib.contextmanager
    def indented(self) -> Iterator[None]:
        self.extra_indent += 1
        yield
        self.extra_indent -= 1
