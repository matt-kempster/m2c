from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = ROOT / "browser" / "dist"
LOCKFILE = ROOT / "browser" / "vendor-lock.json"
M2C_JS = DIST_DIR / "m2c.js"
VENDOR_PATHS_JS = DIST_DIR / "vendor-paths.js"

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


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_file(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def read_vendor_lock() -> dict[str, object]:
    return json.loads(LOCKFILE.read_text(encoding="utf-8"))


def browser_dist_path(relative_path: str) -> str:
    return "./dist/" + relative_path


def vendor_path(lock: dict[str, object], suffix: str) -> str:
    matches = [
        relative_path
        for relative_path in lock["files"]
        if relative_path.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one locked vendor path ending in {suffix}"
        )
    return matches[0]


def write_vendor_paths(lock: dict[str, object]) -> None:
    pyodide_script = vendor_path(lock, "/pyodide.js")
    pyodide_root = pyodide_script.rsplit("/", 1)[0] + "/"
    paths = {
        "pyodideIndexURL": browser_dist_path(pyodide_root),
        "pyodideScript": browser_dist_path(pyodide_script),
        "vizModule": browser_dist_path(vendor_path(lock, "/viz.mjs")),
    }

    print(f"Writing to {VENDOR_PATHS_JS}...")
    VENDOR_PATHS_JS.write_text(
        "window.M2C_VENDOR_PATHS = "
        + json.dumps(paths, ensure_ascii=False, sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )


def write_vendor_lock(lock: dict[str, object]) -> None:
    LOCKFILE.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_vendor_lock() -> None:
    lock = read_vendor_lock()

    for relative_path, metadata in sorted(lock["files"].items()):
        output_path = DIST_DIR / relative_path

        print(f"Downloading {metadata['url']}...")
        data = download_file(metadata["url"])
        metadata["sha256"] = sha256(data)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)

    write_vendor_lock(lock)


def update_vendor_files() -> None:
    lock = read_vendor_lock()

    for relative_path, metadata in sorted(lock["files"].items()):
        output_path = DIST_DIR / relative_path
        expected_hash = metadata["sha256"]

        if output_path.exists():
            data = output_path.read_bytes()
            actual_hash = sha256(data)
            if actual_hash == expected_hash:
                continue
            print(f"Hash mismatch for {output_path}, re-downloading...")

        print(f"Downloading {metadata['url']}...")
        data = download_file(metadata["url"])
        actual_hash = sha256(data)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Hash mismatch for {metadata['url']}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-vendor-lock",
        action="store_true",
        help="download locked vendor URLs and rewrite their SHA-256 hashes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.update_vendor_lock:
        update_vendor_lock()

    update_vendor_files()
    lock = read_vendor_lock()
    write_vendor_paths(lock)

    bundle = {
        str(path.relative_to(ROOT)): path.read_text(encoding="utf-8")
        for path in iter_python_files()
    }

    print(f"Writing to {M2C_JS}...")

    M2C_JS.parent.mkdir(parents=True, exist_ok=True)
    M2C_JS.write_text(
        "window.M2C_PYTHON_FILES = "
        + json.dumps(bundle, ensure_ascii=False, sort_keys=True)
        + ";\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
