#!/usr/bin/env python3
"""Regenerate DATASET_CARD.md from a build_report.json (numbers come from the report only).

  make_dataset_card.py runs/sft/build_report.json [--out card.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gwaya.data_build import render_card  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = json.loads(Path(a.report).read_text())
    card = render_card(rep.get("kind", "sft"), rep)
    Path(a.out or Path(a.report).with_name("DATASET_CARD.md")).write_text(card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
