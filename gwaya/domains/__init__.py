"""Multi-domain tasks and checkers (python, rust, lean4, math) for GwenLaya v4."""
from __future__ import annotations

from gwaya.domains.checkers import check
from gwaya.domains.task import CheckResult, Task

__all__ = ["Task", "CheckResult", "check", "load_tasks"]


def load_tasks(*args, **kwargs):
    from gwaya.domains.loaders import load_tasks as _load
    return _load(*args, **kwargs)
