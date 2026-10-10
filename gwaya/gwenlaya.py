"""GwenLaya: Laya-routed, oracle-gated, calibrated answering (work item T6).

Pipeline per prompt (docs/GWENLAYA_PREREGISTRATION.md section 1):
  1. route: a Laya pre-generation head scores each tier; the cheapest tier with
     p >= tau_route is used, and if none qualifies the result is ABSTAIN.
  2. generate with that tier;
  3. run the domain's gate-visible checker (gwaya.domains.checkers; fail-closed);
  4. a post-generation scorer produces a raw score that a calibrator fitted on the disjoint
     calibration split C maps to p(correct);
  5. verdict, by precedence:
       VERIFIED        every executed gate check passed (p is reported, never a substitute) and the
                       verdict policy below does not demote the pass
       LIKELY_CORRECT  not verified, calibrated p >= tau_hi (never for lean4)
       LIKELY_WRONG    the gate refuted the candidate, or calibrated p <= tau_lo (answer withheld)
       ESCALATE        undecided and a higher tier exists but escalation is not permitted
                       (disabled or budget exhausted); the caller should escalate
       ABSTAIN         undecided and nothing left to try

Honesty notes. Laya itself (router / post head) is injected as plain callables; no Laya weights
are loaded or assumed here. Without a scorer plus a fitted calibration, p_correct is None and
only VERIFIED can be answered (fail-closed). Gate-visible checks only: the caller must not put
scoring/hidden checks in ``checker_payload``. Math has no program-of-thought gate in this repo
yet, so math without a reference answer is UNVERIFIED (the PoT gate is TBD).

Verdict policy (recorded in evidence["policy"] on every output):
  * weak gate (python/rust, OFF by default): with ``min_visible_tests`` configured for the domain, a
    gate pass backed by fewer visible tests (or an unknown count) is not VERIFIED; it takes the same
    calibrator path as an unverified gate. A weak pass never becomes LIKELY_WRONG on the gate alone.
  * lean4 without a non-empty ``formal_statement`` is never VERIFIED: the checker cannot tell the
    task's theorem from an easier one the model wrote (``theorem easy : True := trivial``).
  * lean4 is never LIKELY_CORRECT: the frozen Laya never saw Lean, so only the kernel can vouch.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GATE_STRENGTH_DOMAINS = ("python", "rust")
MIN_VISIBLE_TESTS_ENV = "GWENLAYA_MIN_VISIBLE_TESTS"

from gwaya.domains.checkers import check as domain_check
from gwaya.domains.task import DOMAINS, CheckResult, Task

VERDICTS = ("VERIFIED", "LIKELY_CORRECT", "LIKELY_WRONG", "ABSTAIN", "ESCALATE")
ANSWERED = ("VERIFIED", "LIKELY_CORRECT")
DEFAULT_MAX_NEW_TOKENS = 1024  # plan.json generation_settings: 1024 in every domain
_EPS = 1e-6

# ── calibrators ─────────────────────────────────────────────────────────────


def _logit(s: float) -> float:
    s = min(max(float(s), _EPS), 1.0 - _EPS)
    return math.log(s / (1.0 - s))


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))


class PlattCalibrator:
    """p = sigmoid(a * logit(score) + b), fitted by Newton's method on the log-loss."""
    kind = "platt"

    def __init__(self, a: float = 1.0, b: float = 0.0) -> None:
        self.a, self.b = float(a), float(b)

    def fit(self, scores: Sequence[float], y: Sequence[float]) -> "PlattCalibrator":
        z, t = [_logit(v) for v in scores], [float(v) for v in y]
        if not t or min(t) == max(t):
            m = (sum(t) + 1.0) / (len(t) + 2.0) if t else 0.5  # Laplace-smoothed constant
            self.a, self.b = 0.0, math.log(m / (1.0 - m))
            return self
        mean = sum(t) / len(t)
        a, b, lam = 0.0, math.log(mean / (1.0 - mean)), 1e-6
        for _ in range(100):
            p = [_sigmoid(a * zi + b) for zi in z]
            ga = sum((pi - ti) * zi for pi, ti, zi in zip(p, t, z)) + lam * a
            gb = sum(pi - ti for pi, ti in zip(p, t)) + lam * b
            w = [pi * (1 - pi) for pi in p]
            haa = sum(wi * zi * zi for wi, zi in zip(w, z)) + lam
            hab = sum(wi * zi for wi, zi in zip(w, z))
            hbb = sum(w) + lam
            det = haa * hbb - hab * hab
            da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
            a, b = a - da, b - db
            if max(abs(da), abs(db)) < 1e-9:
                break
        self.a, self.b = a, b
        return self

    def predict(self, scores: Sequence[float] | float) -> list[float]:
        xs = [scores] if isinstance(scores, (int, float)) else scores
        return [_sigmoid(self.a * _logit(v) + self.b) for v in xs]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "a": self.a, "b": self.b}


class IsotonicCalibrator:
    """Non-decreasing map fitted by pool-adjacent-violators, linearly interpolated between block
    means and clamped outside the fit range."""
    kind = "isotonic"

    def __init__(self, xs: Sequence[float] | None = None, ys: Sequence[float] | None = None) -> None:
        self.xs = list(xs or [])
        self.ys = list(ys or [])

    def fit(self, scores: Sequence[float], y: Sequence[float]) -> "IsotonicCalibrator":
        pairs = sorted(zip((float(v) for v in scores), (float(v) for v in y)), key=lambda t: t[0])
        blocks: list[list[float]] = []  # [sum_y, weight, sum_x]
        for si, ti in pairs:
            blocks.append([ti, 1.0, si])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
                top = blocks.pop()
                for k in range(3):
                    blocks[-1][k] += top[k]
        self.xs = [b[2] / b[1] for b in blocks]
        self.ys = [b[0] / b[1] for b in blocks]
        return self

    def _one(self, x: float) -> float:
        xs, ys = self.xs, self.ys
        if not xs:
            return 0.5
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        j = bisect.bisect_right(xs, x)
        x0, x1 = xs[j - 1], xs[j]
        return ys[j - 1] + (ys[j] - ys[j - 1]) * (x - x0) / (x1 - x0)

    def predict(self, scores: Sequence[float] | float) -> list[float]:
        xs = [scores] if isinstance(scores, (int, float)) else scores
        return [self._one(float(v)) for v in xs]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "xs": self.xs, "ys": self.ys}


def calibrator_from_dict(d: Mapping[str, Any]) -> PlattCalibrator | IsotonicCalibrator:
    if d["kind"] == "platt":
        return PlattCalibrator(d["a"], d["b"])
    if d["kind"] == "isotonic":
        return IsotonicCalibrator(d["xs"], d["ys"])
    raise ValueError(f"unknown calibrator kind {d['kind']!r}")


ISOTONIC_MIN_N = 100
CV_FOLDS = 5


def select_calibrator(scores: Sequence[float], y: Sequence[float], method: str = "auto"
                      ) -> tuple[PlattCalibrator | IsotonicCalibrator, dict[str, Any]]:
    """Fit Platt and isotonic on the calibration split and pick one.

    Selection rule (an implementation default; the preregistration names "temperature/isotonic"
    without a rule, so this must be written into the signed addendum before E is generated):
    ``auto`` picks the lower 5-fold cross-validated Brier score on C (folds = index mod 5);
    isotonic is only a candidate when n >= 100; ties go to Platt (fewer parameters).
    ``platt`` / ``isotonic`` force a method.
    """
    s, t = [float(v) for v in scores], [float(v) for v in y]
    n = len(s)
    report: dict[str, Any] = {"n": n, "rule": "min 5-fold CV Brier; isotonic needs n>=100; tie->platt"}
    if method == "platt":
        return PlattCalibrator().fit(s, t), {**report, "chosen": "platt", "forced": True}
    if method == "isotonic":
        return IsotonicCalibrator().fit(s, t), {**report, "chosen": "isotonic", "forced": True}
    if method != "auto":
        raise ValueError(f"unknown method {method!r}")
    cands: dict[str, type] = {"platt": PlattCalibrator}
    if n >= ISOTONIC_MIN_N:
        cands["isotonic"] = IsotonicCalibrator
    cv: dict[str, float] = {}
    if n >= CV_FOLDS * 2:
        for name, cls in cands.items():
            err = 0.0
            for f in range(CV_FOLDS):
                tr = [i for i in range(n) if i % CV_FOLDS != f]
                te = [i for i in range(n) if i % CV_FOLDS == f]
                pred = cls().fit([s[i] for i in tr], [t[i] for i in tr]).predict([s[i] for i in te])
                err += sum((pi - t[i]) ** 2 for pi, i in zip(pred, te))
            cv[name] = err / n
    chosen = min(cv, key=lambda k: (cv[k], k != "platt")) if cv else "platt"
    report.update({"cv_brier": cv, "chosen": chosen, "forced": False})
    return cands[chosen]().fit(s, t), report


def choose_tau_hi(p: Sequence[float], y: Sequence[float], alpha: float = 0.10, min_selected: int = 20) -> float:
    """Split-conformal risk control: the smallest threshold t (largest coverage) whose selected
    set {p >= t} has n >= min_selected and (errors + 1) / (n + 1) <= alpha. +inf if none."""
    for t in sorted(set(p)):
        sel = [yi for pi, yi in zip(p, y) if pi >= t]
        wrong = sum(1 for yi in sel if yi < 0.5)
        if len(sel) >= min_selected and (wrong + 1.0) / (len(sel) + 1.0) <= alpha:
            return float(t)
    return math.inf


def choose_tau_lo(p: Sequence[float], y: Sequence[float], precision: float = 0.90, min_selected: int = 20) -> float:
    """The largest threshold t whose LIKELY_WRONG set {p <= t} has n >= min_selected and
    wrong-precision >= `precision`. -inf if none."""
    for t in sorted(set(p), reverse=True):
        sel = [yi for pi, yi in zip(p, y) if pi <= t]
        wrong = sum(1 for yi in sel if yi < 0.5)
        if len(sel) >= min_selected and wrong / len(sel) >= precision:
            return float(t)
    return -math.inf


@dataclass
class CalibrationArtifact:
    """Frozen calibration for one domain (or "*"). JSON-serialisable; ``sha256`` goes into the
    signed addendum. Thresholds are in calibrated-p space."""
    calibrator: dict[str, Any]
    tau_hi: float
    tau_lo: float
    tau_route: float | None = None
    alpha: float = 0.10
    lo_precision: float = 0.90
    n_cal: int = 0
    selection: dict[str, Any] = field(default_factory=dict)

    def model(self) -> PlattCalibrator | IsotonicCalibrator:
        return calibrator_from_dict(self.calibrator)

    def to_dict(self) -> dict[str, Any]:
        return {"calibrator": self.calibrator, "tau_hi": _jf(self.tau_hi), "tau_lo": _jf(self.tau_lo),
                "tau_route": self.tau_route, "alpha": self.alpha, "lo_precision": self.lo_precision,
                "n_cal": self.n_cal, "selection": self.selection}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CalibrationArtifact":
        return cls(calibrator=dict(d["calibrator"]), tau_hi=_pf(d["tau_hi"]), tau_lo=_pf(d["tau_lo"]),
                   tau_route=d.get("tau_route"), alpha=d.get("alpha", 0.10),
                   lo_precision=d.get("lo_precision", 0.90), n_cal=d.get("n_cal", 0),
                   selection=dict(d.get("selection", {})))

    @property
    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()


def _jf(x: float) -> float | str:
    return "inf" if x == math.inf else "-inf" if x == -math.inf else x


def _pf(x: Any) -> float:
    return {"inf": math.inf, "-inf": -math.inf}.get(x, x) if isinstance(x, str) else float(x)


def fit_calibration(scores: Sequence[float], correct: Sequence[bool | float], *, alpha: float = 0.10,
                    lo_precision: float = 0.90, min_selected: int = 20, method: str = "auto",
                    tau_route: float | None = None) -> CalibrationArtifact:
    """Fit calibrator + tau_hi + tau_lo on the calibration split C ONLY (never on E).
    ``correct`` is the hidden/full-check outcome, not the gate outcome."""
    s = [float(v) for v in scores]
    y = [float(v) for v in correct]
    if len(s) != len(y) or not s:
        raise ValueError("scores and correct must be equal-length non-empty sequences")
    model, report = select_calibrator(s, y, method)
    p = model.predict(s)
    return CalibrationArtifact(
        calibrator=model.to_dict(), tau_hi=choose_tau_hi(p, y, alpha, min_selected),
        tau_lo=choose_tau_lo(p, y, lo_precision, min_selected), tau_route=tau_route,
        alpha=alpha, lo_precision=lo_precision, n_cal=len(s), selection=report)


def save_calibrations(path: str | Path, cals: Mapping[str, CalibrationArtifact]) -> None:
    Path(path).write_text(json.dumps({k: v.to_dict() for k, v in cals.items()}, indent=2, sort_keys=True))


def load_calibrations(path: str | Path) -> dict[str, CalibrationArtifact]:
    raw = json.loads(Path(path).read_text())
    return {k: CalibrationArtifact.from_dict(v) for k, v in raw.items()}


# ── system ──────────────────────────────────────────────────────────────────


@dataclass
class Tier:
    """A generator tier. ``generate(prompt, domain) -> response text``. List order = cheapest first."""
    name: str
    generate: Callable[[str, str], str]


def ollama_tier(model: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> Tier:
    """Greedy (temperature 0) Ollama tier. Network is only touched when the tier is called."""
    from gwaya.generators import OllamaGenerator
    gen = OllamaGenerator(model=model)
    return Tier(model, lambda prompt, domain: gen(prompt, temperature=0.0, max_tokens=max_new_tokens,
                                                  seed=0, domain=domain))


Router = Callable[[str, str], Mapping[str, float]]      # (prompt, domain) -> {tier_name: p_t(correct)}
Scorer = Callable[[dict[str, Any]], float]             # features -> raw score in [0, 1]
Checker = Callable[[Task, str], CheckResult]


def _min_k(domain: str, value: Any) -> int:
    if domain not in GATE_STRENGTH_DOMAINS:
        raise ValueError(f"min_visible_tests: domain {domain!r} not in {GATE_STRENGTH_DOMAINS}")
    if isinstance(value, bool) or not isinstance(value, int):
        try:
            value = int(str(value).strip())
        except ValueError:
            raise ValueError(f"min_visible_tests[{domain}]: {value!r} is not an integer") from None
    if value < 0:
        raise ValueError(f"min_visible_tests[{domain}]: {value} < 0")
    return value


def parse_min_visible_tests(spec: Mapping[str, Any] | str | None) -> dict[str, int]:
    """Normalise the weak-gate policy to {domain: k_min >= 1}; 0 switches a domain off.

    ``spec`` is a mapping, or the env form: an int for python and rust (``"3"``) or ``"python=3,rust=2"``.
    None or an empty string means OFF. Anything malformed raises ValueError (never silently ignored)."""
    if spec is None:
        return {}
    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return {}
        if "=" not in text:
            k = _min_k("python", text)
            out = {d: k for d in GATE_STRENGTH_DOMAINS}
        else:
            out = {}
            for part in text.split(","):
                key, sep, val = part.partition("=")
                key = key.strip()
                if not sep or not key or key in out:
                    raise ValueError(f"{MIN_VISIBLE_TESTS_ENV}: malformed entry {part!r} in {spec!r}")
                out[key] = _min_k(key, val)
    else:
        out = {str(d): _min_k(str(d), v) for d, v in spec.items()}
    return {d: k for d, k in out.items() if k > 0}


class GwenLaya:
    def __init__(self, tiers: Sequence[Tier], router: Router | None = None, scorer: Scorer | None = None,
                 calibrations: Mapping[str, CalibrationArtifact] | None = None,
                 tau_route: float | None = None, budget_s: float | None = None,
                 checker: Checker = domain_check,
                 clock: Callable[[], float] = time.perf_counter,
                 min_visible_tests: Mapping[str, int] | None = None) -> None:
        """``min_visible_tests`` ({'python': k, 'rust': k}) turns on the weak-gate policy: a python/rust gate
        pass backed by fewer than k visible tests (or an unknown count) is not VERIFIED (see the module
        docstring). None reads $GWENLAYA_MIN_VISIBLE_TESTS (an int for both domains, or 'python=3,rust=2');
        unset means OFF. It is OFF by default because no k has been chosen on held-out data: the depth dials
        of A15.2/A17 were measured on E and must not set a default. Malformed values raise ValueError."""
        if not tiers:
            raise ValueError("at least one tier required")
        self.min_visible_tests = parse_min_visible_tests(
            min_visible_tests if min_visible_tests is not None else os.environ.get(MIN_VISIBLE_TESTS_ENV))
        self.tiers = list(tiers)
        self.router, self.scorer = router, scorer
        self.calibrations = dict(calibrations or {})
        self.tau_route = tau_route
        self.budget_s = budget_s
        self.checker, self.clock = checker, clock

    def _cal(self, domain: str) -> CalibrationArtifact | None:
        return self.calibrations.get(domain) or self.calibrations.get("*")

    def _eligible(self, prompt: str, domain: str, cal: CalibrationArtifact | None,
                  ev: dict[str, Any]) -> list[Tier]:
        tau = self.tau_route if self.tau_route is not None else (cal.tau_route if cal else None)
        if self.router is None or tau is None:
            ev["routing"] = "static_ladder" + ("" if self.router else " (no router)")
            return list(self.tiers)
        try:
            probs = dict(self.router(prompt, domain))
        except Exception as exc:  # noqa: BLE001 - router failure must not break answering
            ev["routing"] = f"static_ladder (router_error: {type(exc).__name__})"
            return list(self.tiers)
        ev["routing"] = {"p_tier": probs, "tau_route": tau}
        return [t for t in self.tiers if probs.get(t.name, -math.inf) >= tau]

    def answer(self, prompt: str, domain: str = "python", checker_payload: Mapping[str, Any] | None = None,
               task_id: str = "adhoc", allow_escalation: bool = True) -> dict[str, Any]:
        if domain not in DOMAINS:
            raise ValueError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
        task = Task(domain, task_id, prompt, dict(checker_payload or {}))
        cal = self._cal(domain)
        ev: dict[str, Any] = {"calibrated": cal is not None and self.scorer is not None, "trajectory": [],
                              "policy": self.policy()}
        cost_s = 0.0
        eligible = self._eligible(prompt, domain, cal, ev)
        if not eligible:
            return self._out(None, "ABSTAIN", None, ev, None, 0.0, domain, "no tier met tau_route")
        outcome, last = "UNDECIDED", None
        for i, tier in enumerate(eligible):
            step: dict[str, Any] = {"tier": tier.name}
            ev["trajectory"].append(step)
            t0 = self.clock()
            try:
                response = tier.generate(prompt, domain)
            except Exception as exc:  # noqa: BLE001 - a dead generator is an undecided tier
                response, step["generation_error"] = "", f"{type(exc).__name__}: {exc}"[:300]
            step["gen_s"] = gen_s = max(0.0, self.clock() - t0)
            cost_s += gen_s  # GPU-seconds proxy: generation wall time only (gate runs on CPU)
            last = response
            if "generation_error" in step:
                outcome = "UNDECIDED"
            else:
                res = self.checker(task, response)
                step["gate"] = {"status": res.status, "evidence": res.evidence}
                p = self._p(prompt, domain, tier, i, response, res, ev, step, cal)
                if res.status == "VERIFIED" and not self._demote_pass(task, res, step):
                    return self._out(response, "VERIFIED", p, ev, tier.name, cost_s, domain, None)
                if res.status == "FAILED":
                    outcome = "LIKELY_WRONG"
                    step["reason"] = "gate_refuted_candidate"
                else:  # gate UNVERIFIED, or a pass the policy demoted: the calibrator decides
                    outcome = self._calibrated_outcome(domain, p, cal, step)
                    if outcome == "LIKELY_CORRECT":
                        return self._out(response, "LIKELY_CORRECT", p, ev, tier.name, cost_s, domain, None)
            if i + 1 < len(eligible):
                if allow_escalation and (self.budget_s is None or cost_s < self.budget_s):
                    step["escalated_to"] = eligible[i + 1].name
                    continue
                ev["next_tier"] = eligible[i + 1].name
                ev["escalation_blocked"] = "disabled" if not allow_escalation else "budget_exhausted"
                return self._out(None, "ESCALATE", step.get("p_correct"), ev, tier.name, cost_s, domain, None)
        final = ev["trajectory"][-1]
        verdict = "LIKELY_WRONG" if outcome == "LIKELY_WRONG" else "ABSTAIN"
        return self._out(None, verdict, final.get("p_correct"), ev, eligible[-1].name, cost_s, domain, None)

    def policy(self) -> dict[str, Any]:
        """The verdict policy in force, as written into evidence["policy"]."""
        return {"min_visible_tests": dict(self.min_visible_tests),
                "lean4": {"verified_requires_formal_statement": True, "likely_correct": False}}

    def _demote_pass(self, task: Task, res: CheckResult, step: dict[str, Any]) -> bool:
        """True when a gate pass must not be reported as VERIFIED (the step records why)."""
        k_min = self.min_visible_tests.get(task.domain)
        if k_min is not None:
            n = res.evidence.get("visible_tests")
            step["gate_strength"] = {"visible_tests": n, "min_required": k_min}
            if isinstance(n, bool) or not isinstance(n, int) or n < k_min:
                step["reason"] = "weak_gate"
                return True
        if task.domain == "lean4":
            stmt = task.checker_payload.get("formal_statement")
            if not (isinstance(stmt, str) and stmt.strip()):
                step["reason"] = "lean_no_formal_statement"
                return True
        return False

    @staticmethod
    def _calibrated_outcome(domain: str, p: float | None, cal: CalibrationArtifact | None,
                            step: dict[str, Any]) -> str:
        """LIKELY_CORRECT / LIKELY_WRONG / UNDECIDED for a candidate the gate did not verify, or whose pass the
        policy demoted (shared path). The decision goes to step["decision"]; step["reason"] keeps a demotion
        reason when one is set. A demoted pass can only become LIKELY_WRONG through p <= tau_lo."""
        if p is not None and cal is not None and p >= cal.tau_hi:
            if domain != "lean4":
                step["decision"] = "p_at_or_above_tau_hi"
                return "LIKELY_CORRECT"
            # the frozen Laya never saw Lean: an unchecked proof is never presented as likely correct
            step["decision"] = step["reason"] = "lean_requires_kernel_verification"
            return "UNDECIDED"
        if p is not None and cal is not None and p <= cal.tau_lo:
            decision, outcome = "p_at_or_below_tau_lo", "LIKELY_WRONG"
        else:
            decision, outcome = "gate_unverified_and_p_unavailable_or_between_thresholds", "UNDECIDED"
        step["decision"] = decision
        step.setdefault("reason", decision)
        return outcome

    def _p(self, prompt: str, domain: str, tier: Tier, i: int, response: str, res: CheckResult,
           ev: dict[str, Any], step: dict[str, Any], cal: CalibrationArtifact | None) -> float | None:
        if self.scorer is None or cal is None:
            return None
        try:
            raw = float(self.scorer({"prompt": prompt, "domain": domain, "tier": tier.name, "tier_index": i,
                                     "response": response, "gate_status": res.status,
                                     "gate_evidence": res.evidence}))
            if not (0.0 <= raw <= 1.0) or raw != raw:
                raise ValueError(f"raw score {raw} outside [0, 1]")
        except Exception as exc:  # noqa: BLE001
            step["scorer_error"] = f"{type(exc).__name__}: {exc}"[:300]
            return None
        p = cal.model().predict(raw)[0]
        step["raw_score"], step["p_correct"] = raw, p
        return p

    @staticmethod
    def _out(answer: str | None, verdict: str, p: float | None, ev: dict[str, Any], tier: str | None,
             cost_s: float, domain: str, note: str | None) -> dict[str, Any]:
        if note:
            ev["note"] = note
        ev["escalated"] = len(ev.get("trajectory", [])) > 1
        return {"answer": answer, "verdict": verdict, "p_correct": p, "evidence": ev, "tier": tier,
                "cost_s": round(cost_s, 6), "domain": domain,
                "answered": verdict in ANSWERED}


def payload_for(domain: str, tests: str = "", answer: str = "", formal_statement: str = "",
                min_visible_tests: int | None = None) -> dict[str, Any]:
    """Gate-visible checker payload from loose API fields (empty fields are omitted).

    ``min_visible_tests`` (python/rust only; opt-in, None omits it) is passed to checkers._apply_strength, which
    fails closed (UNVERIFIED) when the visible tests are fewer or cannot be counted. Other domains raise ValueError."""
    key = {"python": ("tests", tests), "rust": ("tests", tests), "math": ("answer", answer),
           "lean4": ("formal_statement", formal_statement)}.get(domain)
    out = {key[0]: key[1]} if key and key[1].strip() else {}
    if min_visible_tests is not None:
        out["min_visible_tests"] = _min_k(domain, min_visible_tests)
    return out


_SYSTEM: GwenLaya | None = None


def get_system() -> GwenLaya:
    """Process-wide lazily built default system (tests replace it via set_system)."""
    global _SYSTEM
    if _SYSTEM is None:
        _SYSTEM = system_from_env()
    return _SYSTEM


def set_system(system: GwenLaya | None) -> None:
    global _SYSTEM
    _SYSTEM = system


def system_from_env() -> GwenLaya:
    """Build the default system from env (no network at construction time).
    GWENLAYA_TIERS: comma-separated Ollama tags, cheapest first (default: the locally installed
    qwen2.5-coder:1.5b,3b,7b; the preregistered Qwen3.5 ladder is not pulled yet).
    GWENLAYA_CALIBRATION: path to a JSON written by save_calibrations. No Laya router/scorer is
    wired by default, so without them only VERIFIED answers are returned."""
    names = [n.strip() for n in os.environ.get(
        "GWENLAYA_TIERS", "qwen2.5-coder:1.5b,qwen2.5-coder:3b,qwen2.5-coder:7b").split(",") if n.strip()]
    cal_path = os.environ.get("GWENLAYA_CALIBRATION")
    cals = load_calibrations(cal_path) if cal_path and Path(cal_path).is_file() else {}
    return GwenLaya([ollama_tier(n) for n in names], calibrations=cals)
