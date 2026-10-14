#!/usr/bin/env python3
"""Lever D2 (A26): cross-tier agreement of candidates on shared inputs as a downstream accept signal.

Reads ONLY dev_table.jsonl (F python + E' python). Never opens a test table.
Inputs shared by all tiers of a task: visible example calls parsed from the PROMPT
(MBPP assert, BigCodeBench '>>> f(...)', LeetCode 'Input: a = .., b = ..', generic 'f(literal args)'),
plus deterministic single-argument mutations, plus (when the prompt has no example) a seeded
pool of generic synthetic arguments filtered to calls that at least one tier executes without error.
Features use only candidate outputs (never the hidden-check label). Labels only enter at scoring.
Deterministic, seed 0. Candidates are executed in rlimit-bounded subprocesses.
"""
import ast, base64, concurrent.futures as cf, copy, hashlib, json, math, os, pickle
import random, re, subprocess, sys, tempfile, zlib, collections as C
import numpy as np

SEED = 0
ROOT = "/mnt/data/home/xavkal/gwaya-data/trust_loop"
DEV = f"{ROOT}/dev_table.jsonl"
OUT = f"{ROOT}/lever_D_consensus_agreement.json"
CACHE = f"{ROOT}/lever_D_exec_cache.json"
PY = sys.executable
T2, T4, T9 = "qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"

# ----------------------------------------------------------------- harness (runs in subprocess)
HARNESS = r'''
import sys, os, pickle, base64, copy, signal, io, resource, re, random, builtins, json
resource.setrlimit(resource.RLIMIT_AS, (1500*1024*1024, 1500*1024*1024))
resource.setrlimit(resource.RLIMIT_CPU, (40, 40))
payload = pickle.loads(base64.b64decode(sys.stdin.buffer.read()))
code, entry, cls_mode, inputs = payload["code"], payload["entry"], payload["cls"], payload["inputs"]
real_stdout = sys.stdout
sys.stdin = open(os.devnull)
builtins.input = lambda *a, **k: (_ for _ in ()).throw(EOFError())
def norm(o, d=0):
    if d > 6: return "<deep>"
    if o is None or isinstance(o, (bool, str)): return o
    if isinstance(o, int): return o
    if isinstance(o, float):
        if o != o: return "nan"
        if o in (float("inf"), float("-inf")): return str(o)
        return float("%.6g" % o)
    if isinstance(o, (list, tuple)): return [norm(x, d+1) for x in o]
    if isinstance(o, (set, frozenset)): return sorted([norm(x, d+1) for x in o], key=repr)
    if isinstance(o, dict): return sorted([[norm(k, d+1), norm(v, d+1)] for k, v in o.items()], key=repr)
    try:
        import numpy as np
        if isinstance(o, np.ndarray): return norm(o.tolist(), d+1)
        if isinstance(o, np.generic): return norm(o.item(), d+1)
    except Exception: pass
    try:
        import pandas as pd
        if isinstance(o, pd.DataFrame): return ["DF", list(map(str, o.columns)), norm(o.values.tolist(), d+1)]
        if isinstance(o, pd.Series): return ["SER", norm(o.tolist(), d+1)]
    except Exception: pass
    r = repr(o)
    return re.sub(r" at 0x[0-9a-fA-F]+", "", r)[:200]
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM, h)
res = []
ns = {"__name__": "cand"}
status = "ok"
try:
    sys.stdout = io.StringIO()
    random.seed(0)
    signal.alarm(10)
    exec(compile(code, "<cand>", "exec"), ns)
    signal.alarm(0)
except BaseException as e:
    signal.alarm(0)
    status = "load:" + type(e).__name__
sys.stdout = io.StringIO()
fn = None
if status == "ok":
    try:
        if cls_mode:
            fn = getattr(ns["Solution"](), entry)
        else:
            fn = ns[entry]
    except BaseException as e:
        status = "noentry:" + type(e).__name__
for args, kwargs in inputs:
    if fn is None:
        res.append("NOFN"); continue
    try:
        a2, k2 = copy.deepcopy(args), copy.deepcopy(kwargs)
        random.seed(0)
        signal.setitimer(signal.ITIMER_REAL, 2.0)
        o = fn(*a2, **k2)
        signal.setitimer(signal.ITIMER_REAL, 0)
        res.append(["OK", norm(o)])
    except TO:
        res.append("TIMEOUT")
    except BaseException as e:
        signal.setitimer(signal.ITIMER_REAL, 0)
        res.append("EXC:" + type(e).__name__)
    sys.stdout = io.StringIO()
sys.stdout = real_stdout
sys.stdout.write(json.dumps({"status": status, "res": res}, default=str))
'''

# ----------------------------------------------------------------- input extraction
SAFE = {"set": set, "frozenset": frozenset, "tuple": tuple, "list": list, "dict": dict, "float": float,
        "int": int, "str": str, "range": range, "True": True, "False": False, "None": None,
        "inf": float("inf"), "complex": complex, "bytes": bytes, "sorted": sorted, "abs": abs}

def safe_eval(node):
    try:
        return True, eval(compile(ast.Expression(node), "<e>", "eval"), {"__builtins__": {}}, dict(SAFE))
    except Exception:
        return False, None

def balanced_calls(text, name):
    for m in re.finditer(r"(?<![\w.])" + re.escape(name) + r"\s*\(", text):
        i = m.end() - 1; depth = 0; j = i; instr = None
        while j < len(text):
            ch = text[j]
            if instr:
                if ch == "\\": j += 1
                elif ch == instr: instr = None
            elif ch in "\"'": instr = ch
            elif ch == "(": depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0: break
            j += 1
        if depth == 0:
            ls = text.rfind("\n", 0, m.start()) + 1
            if text[ls:m.start()].strip().startswith("def "): continue
            yield text[m.start():j + 1]

def parse_call(src):
    try:
        t = ast.parse(src.strip(), mode="eval").body
    except Exception:
        return None
    if not isinstance(t, ast.Call): return None
    args, kwargs = [], {}
    for a in t.args:
        ok, v = safe_eval(a)
        if not ok: return None
        args.append(v)
    for k in t.keywords:
        if k.arg is None: return None
        ok, v = safe_eval(k.value)
        if not ok: return None
        kwargs[k.arg] = v
    return tuple(args), kwargs

LC_START = re.compile(r"class\s+Solution\s*:\s*\n\s*def\s+(\w+)\s*\(self")

def task_spec(prompt, cands):
    """-> (entry, cls_mode, visible_inputs). Prompt-only parsing."""
    m = LC_START.search(prompt)
    vis = []
    if m:
        entry = m.group(1)
        for line in re.findall(r"Input:\s*(.+)", prompt):
            s = re.sub(r"\btrue\b", "True", line); s = re.sub(r"\bfalse\b", "False", s); s = re.sub(r"\bnull\b", "None", s)
            s = s.strip().rstrip(".")
            try:
                t = ast.parse("f(" + s + ")", mode="eval").body
                kw = {}; good = True
                for k in t.keywords:
                    ok, v = safe_eval(k.value)
                    if not ok: good = False; break
                    kw[k.arg] = v
                if good and kw and not t.args: vis.append(((), kw))
            except Exception:
                pass
        return entry, True, vis[:4]
    entry = None
    m = re.search(r"assert\s+(\w+)\s*\(", prompt)
    if m: entry = m.group(1)
    if entry is None:
        ms = re.findall(r"def\s+(\w+)\s*\(", prompt)
        if ms: entry = ms[-1]
    if entry is None:
        for c in cands:
            ms = re.findall(r"^def\s+(\w+)\s*\(", c, flags=re.M)
            if ms: entry = ms[-1]; break
    if entry is None: return None, False, []
    seen = set()
    for src in balanced_calls(prompt, entry):
        pc = parse_call(src)
        if pc is None: continue
        key = repr(pc)
        if key in seen: continue
        seen.add(key); vis.append(pc)
    return entry, False, vis[:4]

def mutate_value(v, rng):
    if isinstance(v, bool): return not v
    if isinstance(v, int): return rng.choice([v + 1, v - 1, 0, v * 2, -v, 7])
    if isinstance(v, float): return rng.choice([v + 1.0, -v, 0.0, v * 2])
    if isinstance(v, str):
        return rng.choice([v[::-1], v.upper(), v[1:], v + v, "", "xyz"])
    if isinstance(v, (list, tuple)):
        L = list(v); r = rng.choice([L[::-1], L[1:], L[:-1], L + L[:1], sorted(L, key=repr), []])
        return tuple(r) if isinstance(v, tuple) else r
    if isinstance(v, dict) and v:
        k = rng.choice(sorted(v, key=repr)); d = dict(v); d.pop(k); return d
    if isinstance(v, (set, frozenset)) and v:
        return type(v)(sorted(v, key=repr)[1:])
    return v

def mutations(vis, tid, n=10):
    rng = random.Random(zlib.crc32(tid.encode()) ^ SEED)
    out = []; seen = {repr(x) for x in vis}; tries = 0
    while len(out) < n and tries < 60 and vis:
        tries += 1
        a, k = rng.choice(vis)
        a = list(copy.deepcopy(a)); k = copy.deepcopy(k)
        slots = [("a", i) for i in range(len(a))] + [("k", kk) for kk in k]
        if not slots: break
        typ, ix = rng.choice(slots)
        if typ == "a": a[ix] = mutate_value(a[ix], rng)
        else: k[ix] = mutate_value(k[ix], rng)
        item = (tuple(a), k)
        if repr(item) in seen: continue
        seen.add(repr(item)); out.append(item)
    return out

POOL = [0, 1, 2, 5, -3, 10, 0.5, 2.0, -1.5, "", "a", "hello world", "abcba", "Hello, World 123", [], [1], [3, 1, 2],
        [1, 2, 2, 3, 5], [5, 4, 3], ["a", "bc"], [[1, 2], [3, 4]], {"a": 1, "b": 2}, True, False, None, (1, 2)]

def extract_code(txt):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", txt, flags=re.S)
    if m: return max(m, key=len)
    m = re.search(r"```(?:python|py)?\s*\n(.*)", txt, flags=re.S)
    return m.group(1) if m else txt

def signature_params(cands, entry, cls):
    for c in cands:
        try:
            tree = ast.parse(extract_code(c))
        except Exception:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == entry:
                ps = [a.arg for a in n.args.args]
                if cls and ps and ps[0] == "self": ps = ps[1:]
                return ps, len(n.args.defaults)
    return None, 0

def synthetic_inputs(cands, entry, cls, tid, n=40):
    ps, nd = signature_params(cands, entry, cls)
    if ps is None: return []
    req = max(len(ps) - nd, 0)
    k = len(ps) if len(ps) <= 4 else req
    rng = random.Random(zlib.crc32(tid.encode()) ^ SEED ^ 99)
    out, seen = [], set()
    for _ in range(n * 3):
        a = tuple(copy.deepcopy(rng.choice(POOL)) for _ in range(k))
        if repr(a) in seen: continue
        seen.add(repr(a)); out.append((a, {}))
        if len(out) >= n: break
    return out

# ----------------------------------------------------------------- execution
def run_candidate(code, entry, cls, inputs):
    payload = base64.b64encode(pickle.dumps(dict(code=code, entry=entry, cls=cls, inputs=inputs))).decode()
    with tempfile.TemporaryDirectory() as td:
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "MPLBACKEND": "Agg", "HOME": td,
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            p = subprocess.run([PY, "-I", "-c", HARNESS], input=payload.encode(), capture_output=True, cwd=td,
                               env=env, timeout=120)
            out = p.stdout.decode(errors="replace")
            j = json.loads(out[out.rfind('{"status"'):]) if '{"status"' in out else None
            if j is None: return {"status": "crash", "res": ["CRASH"] * len(inputs)}
            if len(j["res"]) != len(inputs): return {"status": "short", "res": (j["res"] + ["CRASH"] * len(inputs))[:len(inputs)]}
            return j
        except subprocess.TimeoutExpired:
            return {"status": "walltimeout", "res": ["TIMEOUT"] * len(inputs)}
        except Exception as e:
            return {"status": "err:" + type(e).__name__, "res": ["CRASH"] * len(inputs)}

def jkey(x): return json.dumps(x, sort_keys=True, default=str)

# ----------------------------------------------------------------- stats helpers
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - a) / d, 4), round((c + a) / d, 4)]

def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if y.min() == y.max(): return None
    from scipy.stats import rankdata
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def prec_at_cov(y, s, cov=0.5):
    y = np.asarray(y); s = np.asarray(s, float)
    k = int(math.ceil(cov * len(y)))
    o = np.argsort(-s, kind="stable")[:k]
    kk = int(y[o].sum())
    return dict(n=int(k), k=kk, precision=round(kk / k, 4), wilson=wilson(kk, k), coverage=round(k / len(y), 3))

def pc_curve(y, s, covs=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)):
    return {str(c): prec_at_cov(y, s, c)["precision"] for c in covs}

# ----------------------------------------------------------------- main
AGREE_NAMES = ("agree_all", "agree_vis", "maj_eq", "agree_any", "frac_partners_full", "self_ok_vis", "has_inputs", "nonexc_agree")

def main():
    rows = [json.loads(l) for l in open(DEV)]
    by = C.defaultdict(dict)
    for r in rows: by[(r["set"], r["task_id"])][r["tier"]] = r
    cache = {}
    if os.path.exists(CACHE):
        try: cache = json.load(open(CACHE))
        except Exception: cache = {}
    task_plan = {}; jobs = []
    for key, tiers in sorted(by.items()):
        any_r = next(iter(tiers.values()))
        cands = {t: extract_code(r["candidate"]) for t, r in tiers.items()}
        entry, cls, vis = task_spec(any_r["prompt"], [tiers[t]["candidate"] for t in tiers])
        src = "visible" if vis else "synthetic"
        inputs = list(vis)
        if vis: inputs += mutations(vis, key[1])
        elif entry and not cls: inputs = synthetic_inputs(list(cands.values()), entry, cls, key[1])
        task_plan[key] = dict(entry=entry, cls=cls, n_vis=len(vis), inputs=inputs, src=src, n_in=len(inputs))
        for t, code in cands.items():
            if not entry or not inputs: continue
            h = hashlib.sha1((code + "|" + entry + "|" + repr(cls) + "|" + repr(inputs)).encode()).hexdigest()
            jobs.append((key, t, h, code, entry, cls, inputs))
    todo = {}
    for j in jobs:
        if j[2] not in cache: todo[j[2]] = j
    print(f"tasks {len(by)}  jobs {len(jobs)}  to run {len(todo)}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(run_candidate, j[3], j[4], j[5], j[6]): j for j in todo.values()}
        for n, f in enumerate(cf.as_completed(futs)):
            cache[futs[f][2]] = f.result()
            if n % 100 == 0: print("  done", n, flush=True)
    json.dump(cache, open(CACHE, "w"))
    res = {(key, t): cache[h] for key, t, h, *_ in jobs}

    zero = dict(agree_all=0.0, agree_vis=0.0, agree_loose=0.0, agree_any=0.0, n_agree_full=0, frac_partners_full=0.0,
                maj_eq=0.0, self_ok_vis=0.0, self_ok_all=0.0, nonexc_agree=0.0, partners_agree=0.0)

    def feat_for(key, t, partners):
        plan = task_plan[key]; n_vis = plan["n_vis"]; n_in = plan["n_in"]
        f = dict(has_inputs=0, n_inputs=n_in, n_vis=n_vis)
        partners = [p for p in partners if (key, p) in res]
        if (key, t) not in res or not partners:
            f.update(zero); return f
        own = res[(key, t)]["res"]
        valid = list(range(n_in))
        ok = lambda v: isinstance(v, list)
        if plan["src"] == "synthetic":
            valid = [i for i in valid if any(ok(res[(key, x)]["res"][i]) for x in [t] + partners)]
        if not valid:
            f.update(zero); return f
        f["has_inputs"] = 1
        vis_idx = [i for i in valid if i < n_vis] if plan["src"] == "visible" else valid
        def eq(i, p):
            a, b = own[i], res[(key, p)]["res"][i]
            return float(jkey(a) == jkey(b) and ok(a))
        f["agree_all"] = float(np.mean([eq(i, p) for i in valid for p in partners]))
        f["agree_vis"] = float(np.mean([eq(i, p) for i in vis_idx for p in partners])) if vis_idx else f["agree_all"]
        f["agree_loose"] = float(np.mean([float(jkey(own[i]) == jkey(res[(key, p)]["res"][i])) for i in valid for p in partners]))
        f["agree_any"] = float(np.mean([max(eq(i, p) for p in partners) for i in valid]))
        f["n_agree_full"] = int(sum(all(eq(i, p) for i in valid) for p in partners))
        f["frac_partners_full"] = f["n_agree_full"] / len(partners)
        me = []
        for i in valid:
            c = C.Counter(jkey(res[(key, x)]["res"][i]) for x in [t] + partners if ok(res[(key, x)]["res"][i]))
            if not c or not ok(own[i]): me.append(0.0); continue
            top = max(c.values()); ntop = sum(1 for v in c.values() if v == top)
            me.append(float(c[jkey(own[i])] == top and (ntop == 1 or top > 1)))
        f["maj_eq"] = float(np.mean(me))
        f["self_ok_vis"] = float(np.mean([ok(own[i]) for i in vis_idx])) if vis_idx else 0.0
        f["self_ok_all"] = float(np.mean([ok(own[i]) for i in valid]))
        f["nonexc_agree"] = float(np.mean([eq(i, p) for i in valid for p in partners if ok(own[i])] or [0.0]))
        if len(partners) >= 2:
            f["partners_agree"] = float(np.mean([float(jkey(res[(key, partners[0])]["res"][i]) == jkey(res[(key, partners[1])]["res"][i])) for i in valid]))
        else:
            f["partners_agree"] = f["agree_all"]
        return f

    VAR = {
        "two_tier": lambda t: [x for x in (T2, T4) if x != t],      # test-like: only 2B/4B exist
        "three_tier": lambda t: [x for x in (T2, T4, T9) if x != t],
    }
    table = []
    for key, tiers in sorted(by.items()):
        for t, r in tiers.items():
            d = dict(set=key[0], benchmark=r["benchmark"], task_id=key[1], cluster=r["cluster"], tier=t, y=int(r["correct"]),
                     gate_ok=int(r["gate"] == "verified"), mean_lp=r["mean_logprob"], min_lp=r["min_logprob"],
                     ntok=r["completion_tokens"], trunc=int(r["finish_reason"] != "stop"))
            for vn, pf in VAR.items():
                for a, b in feat_for(key, t, pf(t)).items(): d[f"{vn}__{a}"] = b
            table.append(d)

    out = {"seed": SEED, "n_rows": len(table), "n_tasks": len(by), "agreement": {}, "caveats": []}
    ext = C.Counter()
    for key, p in task_plan.items():
        ext[(next(iter(by[key].values()))["benchmark"], p["src"] if p["n_in"] else "none")] += 1
    out["input_extraction_tasks"] = {f"{a}|{b}": v for (a, b), v in sorted(ext.items())}
    out["exec_status_counts"] = dict(C.Counter(v["status"] for v in res.values()))

    def evaluate(sub, variant):
        if len(sub) < 20: return {"n": len(sub)}
        y = np.array([d["y"] for d in sub])
        P = lambda n: np.array([d[f"{variant}__{n}"] for d in sub], float)
        g = np.array([d["gate_ok"] for d in sub], float)
        lp = np.array([d["mean_lp"] for d in sub], float)
        sc = {"gate": g + 0.0, "mean_lp": lp, "min_lp": np.array([d["min_lp"] for d in sub], float),
              "gate_then_lp": g * 100 + lp,
              "agree_all": P("agree_all"), "agree_vis": P("agree_vis"), "maj_eq": P("maj_eq"),
              "agree_any": P("agree_any"), "frac_partners_full": P("frac_partners_full"),
              "gate_then_agree": g * 100 + P("agree_all") + 0.001 * lp}
        r_ = {"n": len(sub), "base_rate": round(float(y.mean()), 4), "has_inputs_frac": round(float(P("has_inputs").mean()), 3)}
        for n, s in sc.items():
            a = auroc(y, s)
            r_[n] = dict(auroc=None if a is None else round(a, 4), cov50=prec_at_cov(y, s, 0.5), curve=pc_curve(y, s))
        rules = {"gate": g == 1, "agree_full": P("frac_partners_full") == 1,
                 "gate&agree_full": (g == 1) & (P("frac_partners_full") == 1),
                 "gate&agree_all>=0.8": (g == 1) & (P("agree_all") >= 0.8),
                 "gate&agree_full&lp>=-0.1": (g == 1) & (P("frac_partners_full") == 1) & (lp >= -0.1)}
        r_["rules"] = {}
        for rn, m in rules.items():
            k = int(y[m].sum()); n = int(m.sum())
            r_["rules"][rn] = dict(n=n, k=k, precision=None if n == 0 else round(k / n, 4), wilson=wilson(k, n),
                                   coverage=round(n / len(y), 3))
        return r_

    sets = {"ALL": lambda d: True, "F": lambda d: d["set"] == "F", "Eprime": lambda d: d["set"] == "Eprime"}
    for variant in VAR:
        out["agreement"][variant] = {}
        for t in (T4, T2, T9):
            for sn, sf in sets.items():
                out["agreement"][variant][f"{t}|{sn}"] = evaluate([d for d in table if d["tier"] == t and sf(d)], variant)
            for bm in ("LeetCode", "FSB", "BCB", "MBPP"):
                sub = [d for d in table if d["tier"] == t and d["benchmark"] == bm]
                if sub: out["agreement"][variant][f"{t}|{bm}"] = evaluate(sub, variant)

    # ---- grouped-CV learned combination (clusters never split across folds)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold

    def design(sub, variant, with_agree, with_base):
        parts = []
        if with_base:
            parts.append(np.array([[d["gate_ok"], d["mean_lp"], d["min_lp"], math.log1p(d["ntok"]), d["trunc"]] for d in sub], float))
        if with_agree:
            A = np.array([[d[f"{variant}__{n}"] for n in AGREE_NAMES] for d in sub], float)
            gcol = np.array([[d["gate_ok"]] for d in sub], float)
            parts += [A, A[:, :1] * gcol, A[:, 4:5] * gcol]
        return np.hstack(parts)

    def cv_scores(sub, variant, with_agree, with_base):
        X = design(sub, variant, with_agree, with_base)
        y = np.array([d["y"] for d in sub]); groups = np.array([d["cluster"] for d in sub])
        oof = np.zeros(len(y)); fp = np.zeros(len(y), bool)
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, random_state=SEED))
            m.fit(X[tr], y[tr])
            th = np.quantile(m.predict_proba(X[tr])[:, 1], 0.5)   # threshold from TRAIN fold only
            p = m.predict_proba(X[te])[:, 1]
            oof[te] = p; fp[te] = p >= th
        return y, oof, fp

    out["learned_grouped_cv"] = {}
    for variant in VAR:
        out["learned_grouped_cv"][variant] = {}
        for t in (T4, T2, T9):
            sub = [d for d in table if d["tier"] == t]
            bms = np.array([d["benchmark"] for d in sub]); e = {}
            for name, wa, wb in (("base_only", False, True), ("agree_only", True, False), ("base+agree", True, True)):
                y, oof, fp = cv_scores(sub, variant, wa, wb)
                n = int(fp.sum()); k = int(y[fp].sum())
                thrv = np.quantile(oof, 0.5)
                bb = {}
                for bm in sorted(set(bms)):
                    m = (bms == bm) & (oof >= thrv); nn = int(m.sum()); kk = int(y[m].sum())
                    bb[bm] = dict(n=nn, precision=None if nn == 0 else round(kk / nn, 4), wilson=wilson(kk, nn),
                                  coverage=round(nn / int((bms == bm).sum()), 3))
                e[name] = dict(auroc=round(auroc(y, oof), 4), oof_cov50=prec_at_cov(y, oof, 0.5), curve=pc_curve(y, oof),
                               train_thr_cov50=dict(n=n, k=k, precision=None if n == 0 else round(k / n, 4), wilson=wilson(k, n),
                                                    coverage=round(n / len(y), 3)),
                               by_benchmark_at_pooled_thr=bb)
            out["learned_grouped_cv"][variant][t] = e

    out["caveats"] = [
        "Agreement is computed on dev tiers only; the sealed test has 2B and 4B only, so the two_tier variant (4B candidate vs 2B partner) is the relevant one.",
        "Inputs are visible prompt examples plus single-argument mutations; mutated LeetCode inputs may violate stated constraints, where correct solutions can legitimately differ.",
        "Tasks without an example (mostly FSB) use synthetic generic arguments filtered to calls that at least one tier runs; weaker evidence.",
        "Two tiers sharing a misunderstanding agree and are both wrong: agreement is not correctness.",
        "Hidden-check labels are used only for scoring and for fitting the CV logistic model on train folds grouped by cluster.",
        "Dev set is 407+169 tasks, single seed; Wilson CIs ignore feature-design selection on the same dev data.",
    ]
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    print("extraction", out["input_extraction_tasks"]); print("status", out["exec_status_counts"])
    for variant in VAR:
        print("==", variant)
        for t in (T4, T2, T9):
            e = out["agreement"][variant][f"{t}|ALL"]
            print(t, "n", e["n"], "base", e["base_rate"], "has_in", e["has_inputs_frac"])
            for n in ("gate", "mean_lp", "gate_then_lp", "agree_all", "maj_eq", "gate_then_agree"):
                print("   %-16s auroc %s cov50 %s" % (n, e[n]["auroc"], e[n]["cov50"]["precision"]))
            for rn, v in e["rules"].items(): print("   rule %-26s %s" % (rn, v))
            for name, v in out["learned_grouped_cv"][variant][t].items():
                print("   cv %-12s auroc %s oof50 %s %s trainthr %s" % (name, v["auroc"], v["oof_cov50"]["precision"], v["oof_cov50"]["wilson"], v["train_thr_cov50"]))

if __name__ == "__main__":
    main()
