from __future__ import annotations

import contextlib
import gc
import io
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

from . import main as main_module
from .main import parse_flags


class BrowserResult(TypedDict):
    returncode: int
    output: str
    output_type: str


_FUNCTION_START_RE = re.compile(
    r"^\s*(?:"
    r"(?:glabel|dlabel|arm_func_start|thumb_func_start|"
    r"non_word_aligned_thumb_func_start|ARM_FUNC_START|THUMB_FUNC_START|"
    r"NON_WORD_ALIGNED_THUMB_FUNC_START)\s+\S+|"
    r"\.fn\s+\S+|"
    r"[A-Za-z_.$][\w.$]*:"
    r")",
    re.MULTILINE,
)
_RECURSION_LIMIT_SET = False


def _ensure_recursion_limit() -> None:
    global _RECURSION_LIMIT_SET
    if not _RECURSION_LIMIT_SET:
        sys.setrecursionlimit(min(2**31 - 1, 10 * sys.getrecursionlimit()))
        _RECURSION_LIMIT_SET = True


def decompile(
    source: str, context: Optional[str] = None, flags: Optional[List[str]] = None
) -> BrowserResult:
    """Run m2c from string inputs, for Pyodide/browser use."""
    if not _FUNCTION_START_RE.search(source):
        source = "glabel foo\n" + source

    stdout = io.StringIO()
    stderr = io.StringIO()
    is_visualize = False
    was_gc_enabled = gc.isenabled()
    gc_thresholds = gc.get_threshold()

    try:
        with tempfile.TemporaryDirectory(prefix="m2c-browser-") as tmpdir:
            base_path = Path(tmpdir)
            asm_path = base_path / "input.s"
            asm_path.write_text(source, encoding="utf-8")

            argv = list(flags or [])
            argv.append("--no-cache")

            if context:
                context_path = base_path / "context.c"
                context_path.write_text(context, encoding="utf-8")
                argv.extend(["--context", str(context_path)])

            argv.append(str(asm_path))

            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                options = parse_flags(argv)
                is_visualize = options.visualize_flowgraph is not None
                _ensure_recursion_limit()
                if options.disable_gc:
                    gc.disable()
                    gc.set_threshold(0)
                returncode = main_module.run(options, visualize_as_dot=True)
    except SystemExit as exc:
        is_visualize = False
        returncode = int(exc.code) if isinstance(exc.code, int) else 1
    except Exception as exc:
        is_visualize = False
        returncode = 1
        stdout.write(f"Internal browser wrapper error:\n{exc}\n")
    finally:
        gc.set_threshold(*gc_thresholds)
        if was_gc_enabled:
            gc.enable()
        else:
            gc.disable()
    err = stderr.getvalue()
    if err:
        stdout.write(err)

    return {
        "returncode": returncode,
        "output": stdout.getvalue(),
        "output_type": "dot" if is_visualize else "text",
    }


def decompile_from_options(options: Dict[str, object]) -> BrowserResult:
    source = str(options.get("source", ""))
    context = options.get("context")
    flags = options.get("flags")

    return decompile(
        source,
        str(context) if context else None,
        [str(flag) for flag in flags] if isinstance(flags, list) else None,
    )


def decompile_from_json(options_json: str) -> BrowserResult:
    options = json.loads(options_json)
    if not isinstance(options, dict):
        return {
            "returncode": 1,
            "output": "Expected browser options JSON to decode to an object.\n",
            "output_type": "text",
        }
    return decompile_from_options(options)
