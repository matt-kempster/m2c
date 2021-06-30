import contextlib
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Union


@dataclass
class CodingStyle:
    newline_after_function: bool
    newline_after_if: bool
    newline_before_else: bool
    pointer_style_left: bool


@dataclass
class Options:
    filenames: List[str]
    function_indexes_or_names: List[Union[int, str]]
    debug: bool
    void: bool
    ifs: bool
    andor_detection: bool
    skip_casts: bool
    reg_vars: List[str]
    goto_patterns: List[str]
    stop_on_error: bool
    print_assembly: bool
    visualize_flowgraph: bool
    c_context: Optional[str]
    dump_typemap: bool
    pdb_translate: bool
    preproc_defines: Dict[str, int]
    coding_style: CodingStyle
    sanitize_tracebacks: bool
    valid_syntax: bool
    emit_globals: bool

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
    pointer_style_left=False,
)


@dataclass
class Formatter:
    coding_style: CodingStyle = DEFAULT_CODING_STYLE
    indent_step: str = " " * 4
    skip_casts: bool = False
    extra_indent: int = 0
    debug: bool = False
    valid_syntax: bool = False

    def indent(self, line: str, indent: int = 0) -> str:
        return self.indent_step * max(indent + self.extra_indent, 0) + line

    @contextlib.contextmanager
    def indented(self) -> Iterator[None]:
        try:
            self.extra_indent += 1
            yield
        finally:
            self.extra_indent -= 1
