from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "browser" / "m2c.generated.js"

INCLUDE_DIRS = (
    ROOT / "m2c",
    ROOT / "m2c_pycparser",
    ROOT / "m2c_pycparser" / "ply",
)


def iter_python_files() -> list[Path]:
    files: list[Path] = []
    for directory in INCLUDE_DIRS:
        files.extend(sorted(directory.glob("*.py")))
    return files


def main() -> None:
    bundle = {
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
        for path in iter_python_files()
    }

    print(f"Writing to {OUTPUT}...")

    OUTPUT.write_text(
        "window.M2C_PYTHON_FILES = "
        + json.dumps(bundle, ensure_ascii=False, sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
