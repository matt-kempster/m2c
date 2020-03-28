from typing import Dict, List

import attr


@attr.s
class CodingStyle:
    newline_after_function: bool = attr.ib()
    newline_after_if: bool = attr.ib()
    newline_before_else: bool = attr.ib()


@attr.s
class Options:
    filename: str = attr.ib()
    debug: bool = attr.ib()
    void: bool = attr.ib()
    ifs: bool = attr.ib()
    andor_detection: bool = attr.ib()
    goto_patterns: List[str] = attr.ib()
    rodata_files: List[str] = attr.ib()
    stop_on_error: bool = attr.ib()
    print_assembly: bool = attr.ib()
    visualize_flowgraph: bool = attr.ib()
    c_context: str = attr.ib()
    preproc_defines: Dict[str, int] = attr.ib()
    coding_style: CodingStyle = attr.ib()
