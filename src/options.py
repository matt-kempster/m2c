import contextlib
from dataclasses import dataclass
import enum
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union


class ChoicesEnum(enum.Enum):
    """A helper class that is easier to use with argparse"""

    def __str__(self) -> str:
        return str(self.value)


@dataclass
class CodingStyle:
    class CommentStyle(ChoicesEnum):
        MULTILINE = "multiline"
        ONELINE = "oneline"
        NONE = "none"

    newline_after_function: bool
    newline_after_if: bool
    newline_before_else: bool
    pointer_style_left: bool
    unknown_underscore: bool
    hex_case: bool
    comment_style: CommentStyle
    comment_column: int


@dataclass
class Options:
    class CompilerEnum(ChoicesEnum):
        IDO = "ido"
        GCC = "gcc"

    class GlobalDeclsEnum(ChoicesEnum):
        ALL = "all"
        USED = "used"
        NONE = "none"

    filenames: List[str]
    function_indexes_or_names: List[Union[int, str]]
    debug: bool
    void: bool
    ifs: bool
    switch_detection: bool
    andor_detection: bool
    skip_casts: bool
    reg_vars: List[str]
    goto_patterns: List[str]
    stop_on_error: bool
    print_assembly: bool
    visualize_flowgraph: bool
    c_contexts: List[Path]
    use_cache: bool
    dump_typemap: bool
    pdb_translate: bool
    preproc_defines: Dict[str, int]
    coding_style: CodingStyle
    sanitize_tracebacks: bool
    valid_syntax: bool
    global_decls: GlobalDeclsEnum
    compiler: CompilerEnum
    print_stack_structs: bool
    unk_inference: bool
    passes: int
    incbin_dir: Optional[Path]

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
    unknown_underscore=False,
    hex_case=False,
    comment_style=CodingStyle.CommentStyle.MULTILINE,
    comment_column=52,
)


@dataclass
class Formatter:
    coding_style: CodingStyle = DEFAULT_CODING_STYLE
    indent_step: str = " " * 4
    skip_casts: bool = False
    extra_indent: int = 0
    debug: bool = False
    valid_syntax: bool = False
    line_length: int = 80

    def indent(self, line: str, indent: int = 0) -> str:
        return self.indent_step * max(indent + self.extra_indent, 0) + line

    @contextlib.contextmanager
    def indented(self) -> Iterator[None]:
        try:
            self.extra_indent += 1
            yield
        finally:
            self.extra_indent -= 1

    def format_array(self, elements: List[str]) -> str:
        # If there are no newlines & the output would be short, put it all on one line.
        # Here, "line length" is just used as a rough guideline: we aren't accounting
        # for the LHS of the assignment or any indentation.
        if not any("\n" in el or len(el) > self.line_length for el in elements):
            output = f"{{ {', '.join(elements)} }}"
            if len(output) < self.line_length:
                return output

        # Otherwise, put each element on its own line (and include a trailing comma)
        output = "{\n"
        for el in elements:
            # Add 1 indentation level to the string
            el = el.replace("\n", "\n" + self.indent_step)
            output += self.indent(f"{el},\n", 1)
        output += "}"

        return output

    def with_comments(self, line: str, comments: List[str], *, indent: int = 0) -> str:
        """Indent `line` and append a list of `comments` joined with ';'"""
        base = self.indent(line, indent=indent)
        # If `comments` is empty; fall back to `Formatter.indent()` behavior
        if (
            not comments
            or self.coding_style.comment_style == CodingStyle.CommentStyle.NONE
        ):
            return base

        # Add padding to the style's `comment_column`, only if `line` is non-empty
        padding = ""
        if line:
            padding = max(1, self.coding_style.comment_column - len(base)) * " "
        if self.coding_style.comment_style == CodingStyle.CommentStyle.ONELINE:
            comment = f"// {'; '.join(comments)}"
        else:
            comment = f"/* {'; '.join(comments)} */"
        return f"{base}{padding}{comment}"

    def format_hex(self, val: int) -> str:
        return format(val, "x").upper()

    def format_int(self, val: int) -> str:
        if abs(val) < 10:
            return str(val)

        return hex(val).upper().replace("X", "x")
