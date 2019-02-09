from typing import Dict

import attr


@attr.s
class Options:
    filename: str = attr.ib()
    debug: bool = attr.ib()
    ifs: bool = attr.ib()
    andor_detection: bool = attr.ib()
    stop_on_error: bool = attr.ib()
    print_assembly: bool = attr.ib()
    visualize_flowgraph: bool = attr.ib()
    preproc_defines: Dict[str, int] = attr.ib()
