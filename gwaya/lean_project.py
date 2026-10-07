"""
gwaya/lean_project.py
=====================
Discovery of a pinned Lean 4 + Mathlib lake project for Mathlib-dependent checks
(miniF2F, ProofNet, Lean-Workbook).

The project is never run through `lake`: lake may write to the project (manifest,
build dir), which must stay read-only inside the sandbox. Instead we resolve the
pinned toolchain's `lean` binary and the package `.olean` directories ourselves and
hand them over as LEAN_PATH.

Layout (see scripts/setup_lean_mathlib.sh):
    <root>/project/            lakefile.lean, lean-toolchain, lake-manifest.json, .lake/
    <root>/elan/toolchains/    the pinned toolchain (kept off the root disk)
    <root>/project/gwenlaya_lean_project.json   recorded pins (written by the setup script)

The project is located via the `project_dir` argument or $GWAYA_LEAN_MATHLIB_DIR.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ENV = "GWAYA_LEAN_MATHLIB_DIR"
META_FILE = "gwenlaya_lean_project.json"
DEFAULT_ROOT = os.path.join(os.environ.get("GWAYA_DATA_ROOT", os.path.expanduser("~/gwaya-data")), "lean-mathlib")


@dataclass(frozen=True)
class LeanProject:
    project_dir: Path
    lean_bin: Path
    toolchain: str
    lean_path: tuple[str, ...]
    ro_binds: tuple[str, ...]
    mathlib_rev: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def lean_path_env(self) -> str:
        return ":".join(self.lean_path)


def _toolchain_dir_name(toolchain: str) -> str:
    # "leanprover/lean4:v4.7.0" -> "leanprover--lean4---v4.7.0" (elan's directory scheme)
    return toolchain.replace("/", "--").replace(":", "---")


def _toolchain_roots(project_dir: Path) -> list[Path]:
    roots = [project_dir.parent / "elan"]
    if os.environ.get("ELAN_HOME"):
        roots.append(Path(os.environ["ELAN_HOME"]))
    roots.append(Path.home() / ".elan")
    return [r / "toolchains" for r in roots]


def _mathlib_rev(project_dir: Path) -> str | None:
    try:
        manifest = json.loads((project_dir / "lake-manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    for pkg in manifest.get("packages", []):
        if pkg.get("name") == "mathlib":
            return pkg.get("rev")
    return None


def default_project_dir() -> Path | None:
    """$GWAYA_LEAN_MATHLIB_DIR if set, else the standard location if it exists."""
    env = os.environ.get(PROJECT_ENV)
    if env:
        return Path(env)
    default = Path(DEFAULT_ROOT) / "project"
    return default if default.is_dir() else None


def discover(project_dir: str | os.PathLike | None = None) -> LeanProject | None:
    """Return the project, or None unless toolchain *and* built Mathlib oleans are present."""
    base = Path(project_dir) if project_dir else default_project_dir()
    if base is None:
        return None
    base = base.resolve()
    try:
        toolchain = (base / "lean-toolchain").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    lean_bin = None
    toolchain_root = None
    for root in _toolchain_roots(base):
        cand = root / _toolchain_dir_name(toolchain)
        if (cand / "bin" / "lean").is_file():
            lean_bin, toolchain_root = (cand / "bin" / "lean").resolve(), cand.resolve()
            break
    if lean_bin is None:
        return None
    packages = base / ".lake" / "packages"
    mathlib_lib = packages / "mathlib" / ".lake" / "build" / "lib"
    if not (mathlib_lib / "Mathlib.olean").is_file():
        return None
    lib_dirs = sorted(
        str(p / ".lake" / "build" / "lib") for p in packages.iterdir() if (p / ".lake" / "build" / "lib").is_dir()
    )
    own = base / ".lake" / "build" / "lib"
    if own.is_dir():
        lib_dirs.append(str(own))
    meta: dict = {}
    try:
        meta = json.loads((base / META_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    # Bind the project and the toolchain (real paths, since packages may be symlinks).
    binds = [str(base), str(toolchain_root)]
    return LeanProject(
        project_dir=base,
        lean_bin=lean_bin,
        toolchain=toolchain,
        lean_path=tuple(lib_dirs),
        ro_binds=tuple(dict.fromkeys(binds)),
        mathlib_rev=_mathlib_rev(base),
        meta=meta,
    )
