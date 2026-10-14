#!/usr/bin/env python3
"""Create a NEW VERSION of an existing Zenodo record from a release PDF. DRAFT ONLY unless --publish is given.

  python3 scripts/release/zenodo_new_version.py                 # creates/uploads a draft, prints the draft URL
  python3 scripts/release/zenodo_new_version.py --publish ID    # publishes that draft (irreversible: a DOI cannot be deleted)

Token: $ZENODO_TOKEN or ~/.config/zenodo/token (scopes deposit:write and deposit:actions). It is read here and sent only
to zenodo.org; it is never printed. Default target: the concept of record 23121788 (GWAYA v3.1, creator Callens).
Use --sandbox to rehearse against sandbox.zenodo.org with a sandbox token.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = "3.12.0"
PDF = ROOT / "papers" / "gwenlaya_v4.pdf"
RELEASE_URL = "https://github.com/xaviercallens/GWAYA-GwenLayaOpenModel/releases/tag/v3.12.0"
DESCRIPTION = """\
<p><b>GwenLaya v4 (revision 3.12.0)</b>: a pre-registered study of answering only when a check can back the answer.
Qwen3.5 generators (2B/4B/9B) with the GWAYA fail-closed gate and the Laya router/calibrator, for Python, math and Lean 4.</p>
<p>New in this revision (phase 2, exploratory, decision rules written before each result): a feedback-retrofitted Laya v2
(no net gain), best-of-8 sampling with the visible gate on 2B/4B/9B (not a lever), a learned selector over sampled pools
(not a lever), and a sealed confirmation on the previously seen, easy set E of a frozen downstream accept policy
(9B: coverage 0.73, precision 0.886, Wilson 95% 0.850 to 0.914). On the fresh set F the same goal is met only at low
coverage; the goal redefinition is post hoc and labelled so. The revision was recomputed by an independent audit
(docs/CLAIM_AUDIT_3120.md in the repository, 8 slips fixed).</p>
<p>Code, data, results and the audit: <a href="https://github.com/xaviercallens/GWAYA-GwenLayaOpenModel">GitHub</a>.</p>
"""


def token(sandbox: bool) -> str:
    t = os.environ.get("ZENODO_TOKEN", "").strip()
    if not t:
        p = Path.home() / ".config" / ("zenodo/sandbox_token" if sandbox else "zenodo/token")
        t = p.read_text().strip() if p.is_file() else ""
    if not t:
        sys.exit("no Zenodo token (ZENODO_TOKEN or ~/.config/zenodo/token)")
    return t


class Z:
    def __init__(self, base: str, tok: str):
        self.base, self.tok = base.rstrip("/"), tok

    def call(self, method: str, url: str, body=None, raw: bytes | None = None, ctype="application/json"):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(url if url.startswith("http") else self.base + url, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.tok}", "Content-Type": ctype})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                txt = r.read()
                return json.loads(txt) if txt else {}
        except urllib.error.HTTPError as e:
            sys.exit(f"{method} {url} -> HTTP {e.code}: {e.read().decode()[:400]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="23121788", help="any record id of the concept to version")
    ap.add_argument("--license", default="cc-by-4.0", help="Zenodo license id (default matches the existing record)")
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--draft", metavar="DRAFT_ID", help="reuse this existing draft (replace its file and metadata) instead of creating a new version")
    ap.add_argument("--publish", metavar="DRAFT_ID", help="publish this draft deposition (irreversible)")
    a = ap.parse_args()
    z = Z("https://sandbox.zenodo.org/api" if a.sandbox else "https://zenodo.org/api", token(a.sandbox))

    if a.publish:
        d = z.call("POST", f"/deposit/depositions/{a.publish}/actions/publish")
        print("PUBLISHED", d.get("doi"), d.get("links", {}).get("record_html"))
        return 0

    if not PDF.is_file():
        sys.exit(f"missing {PDF}")
    sha = hashlib.sha256(PDF.read_bytes()).hexdigest()
    if a.draft:
        draft = z.call("GET", f"/deposit/depositions/{a.draft}")
    else:
        new = z.call("POST", f"/deposit/depositions/{a.record}/actions/newversion")
        draft = z.call("GET", new["links"]["latest_draft"])
    dep_id = draft["id"]
    for f in draft.get("files", []):  # a new version starts with the old files: replace them
        z.call("DELETE", f["links"]["self"])
    bucket = draft["links"]["bucket"]
    name = f"gwenlaya_v4_rev{VERSION}.pdf"
    z.call("PUT", f"{bucket}/{name}", raw=PDF.read_bytes(), ctype="application/octet-stream")
    old = draft.get("metadata", {})
    meta = {
        "title": "GwenLaya v4: a pre-registered study of answering only when a check can back the answer",
        "upload_type": "publication", "publication_type": "preprint",
        "description": DESCRIPTION, "version": VERSION, "access_right": "open", "license": a.license,
        "creators": old.get("creators") or [{"name": "Callens, Xavier", "affiliation": "AutoevolveAI / ANSE"}],
        "keywords": ["GWAYA", "GwenLaya", "fail-closed verification", "LLM code generation", "Lean 4",
                     "calibration", "selective prediction", "Qwen3.5", "TPU"],
        "related_identifiers": [{"relation": "isSupplementedBy", "identifier": RELEASE_URL, "scheme": "url"}],
    }
    z.call("PUT", f"/deposit/depositions/{dep_id}", body={"metadata": meta})
    print(json.dumps({"draft_id": dep_id, "draft_html": draft["links"].get("html"), "file": name, "sha256": sha,
                      "license": a.license, "next": f"python3 {sys.argv[0]} --publish {dep_id}"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
