"""Build the C++ extension in-place without touching pyproject.toml.

The repo's pyproject.toml uses poetry-specific keys under `[project]` that
PEP 621 (and therefore modern setuptools) rejects. We don't want to migrate
the project metadata, so this wrapper temporarily moves pyproject.toml aside
while invoking setup.py, then puts it back — even if the build is interrupted.

Usage:
    python scripts/build_ext.py                 # build_ext --inplace
    python scripts/build_ext.py --force         # force-rebuild
    python scripts/build_ext.py clean           # clean build artefacts
"""
from __future__ import annotations

import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "pyproject.toml"
BAK = ROOT / "pyproject.toml.build_bak"


@contextmanager
def hide_pyproject():
    moved = False
    if PROJ.exists():
        if BAK.exists():
            raise SystemExit(
                f"refusing to start: {BAK.name} already exists — "
                "a previous build was likely interrupted; restore it manually"
            )
        PROJ.rename(BAK)
        moved = True
    try:
        yield
    finally:
        if moved and BAK.exists():
            BAK.rename(PROJ)


def main() -> None:
    args = sys.argv[1:]
    if not args:
        args = ["build_ext", "--inplace"]
    elif args[0] == "clean":
        args = ["clean", "--all"]

    with hide_pyproject():
        completed = subprocess.run(
            [sys.executable, "setup.py", *args],
            cwd=ROOT,
        )
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
