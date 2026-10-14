#!/usr/bin/env python3
"""Lever D1 (A26 loop): downstream trust via candidate audit + differential execution.

Reads ONLY dev_table.jsonl (F python + E' python). Never opens test_table.jsonl.
Stages:  extract  -> per-task example inputs (from the prompt text) + type-preserving mutants
         exec     -> run every tier's candidate in the bwrap sandbox on those inputs (cached)
         features -> static audit + example pass + crash/constant + cross-tier agreement
         eval     -> cluster-grouped CV (GroupKFold by source problem), precision vs coverage
Deterministic, seed 0. No feature uses 'correct' (label only).
"""
import ast, copy, difflib, hashlib, json, math, os, random, re, sys, collections as C
from concurrent.futures import ThreadPoolExecutor
import numpy as np

ROOT = "/home/xavkal/xdev/GWAYA-GwenLayaOpenModel"
sys.path.insert(0, ROOT)
DEV = "/mnt/data/home/xavkal/gwaya-data/trust_loop/dev_table.jsonl"
OUT = "/mnt/data/home/xavkal/gwaya-data/trust_loop/lever_D_gate_strength_stub.json"
CACHE = "/mnt/data/home/xavkal/gwaya-data/trust_loop/lever_D_exec_cache.json"
SEED = 0
MAX_EX = 4          # prompt examples used
N_MUT = 10          # mutants per task
TIERS = ["qwen3.5-2b-bf16", "qwen3.5-4b-bf16", "qwen3.5-9b-bf16"]

# ----------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - a) / d, (c + a) / d]

def code_of(text):
    m = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if m: return max(m, key=len)
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text, re.S)   # unterminated fence
    return m.group(1) if m else text

def lit(node):
    return ast.literal_eval(node)

def _try_eval_text(s):
    try: return True, ast.literal_eval(s.strip())
    except Exception: return False, None

def canon_py(v):
    return v

# ----------------------------------------------------------------- extraction
def prompt_func_names(prompt):
    code = "\n".join(re.findall(r"```python\s*\n(.*?)```", prompt, re.S)) or prompt
    names = []
    try:
        t = ast.parse(code)
        for n in t.body:
            if isinstance(n, ast.FunctionDef): names.append(("f", n.name, [a.arg for a in n.args.args]))
            if isinstance(n, ast.ClassDef):
                for m in n.body:
                    if isinstance(m, ast.FunctionDef) and not m.name.startswith("__"):
                        names.append(("m", n.name + "." + m.name, [a.arg for a in m.args.args if a.arg != "self"]))
    except SyntaxError:
        cm = re.search(r"^class\s+(\w+)", code, re.M)
        for m in re.finditer(r"^\s*def\s+(\w+)\s*\(([^)]*)\)", code, re.M):
            if cm and m.group(0)[0] in " \t" and not m.group(1).startswith("__"):
                names.append(("m", cm.group(1) + "." + m.group(1), [a.split(":")[0].split("=")[0].strip() for a in m.group(2).split(",") if a.strip() and a.strip() != "self"]))
                continue
            names.append(("f", m.group(1), [a.split(":")[0].split("=")[0].strip() for a in m.group(2).split(",") if a.strip() and a.strip() != "self"]))
    return names

def call_to_input(call):
    """ast.Call -> (name, args, kwargs) with literal args or None"""
    try:
        args = [lit(a) for a in call.args]
        kw = {k.arg: lit(k.value) for k in call.keywords if k.arg}
        if len(kw) != len(call.keywords): return None
        name = call.func.id if isinstance(call.func, ast.Name) else (call.func.attr if isinstance(call.func, ast.Attribute) else None)
        return name, args, kw
    except Exception:
        return None

def extract_examples(prompt):
    """returns list of dict(name,args,kw,has_exp,exp) from asserts, doctests, 'Input:/Output:' lines"""
    ex = []
    lines = prompt.split("\n")
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("assert "):
            try:
                t = ast.parse(s)
                cmp_ = t.body[0].test
                if isinstance(cmp_, ast.Compare) and isinstance(cmp_.ops[0], ast.Eq) and isinstance(cmp_.left, ast.Call):
                    ci = call_to_input(cmp_.left)
                    if ci:
                        try: e = lit(cmp_.comparators[0]); he = True
                        except Exception: e, he = None, False
                        ex.append(dict(name=ci[0], args=ci[1], kw=ci[2], has_exp=he, exp=e))
            except Exception: pass
        elif s.startswith(">>> "):
            try:
                t = ast.parse(s[4:], mode="eval").body
                if isinstance(t, ast.Call):
                    ci = call_to_input(t)
                    if ci:
                        he, e = False, None
                        if i + 1 < len(lines) and not lines[i + 1].strip().startswith(">>>"):
                            he, e = _try_eval_text(lines[i + 1])
                        ex.append(dict(name=ci[0], args=ci[1], kw=ci[2], has_exp=he, exp=e))
            except Exception: pass
        elif s.startswith("Input:") or s.startswith("**Input:**") or s.startswith("Input :"):
            body = re.sub(r"^\**Input\**\s*:\**\s*", "", s)
            try:
                t = ast.parse("f(" + body + ")", mode="eval").body
                ci = call_to_input(t)
                if ci and not ci[1]:
                    he, e = False, None
                    for j in range(i + 1, min(i + 4, len(lines))):
                        mo = re.match(r"\**Output\**\s*:\**\s*(.*)$", lines[j].strip())
                        if mo:
                            txt = mo.group(1).strip().replace("true", "True").replace("false", "False").replace("null", "None")
                            he, e = _try_eval_text(txt); break
                    ex.append(dict(name=None, args=[], kw=ci[2], has_exp=he, exp=e))
            except Exception: pass
    return ex

def mutate_value(v, rng):
    if isinstance(v, bool): return (not v)
    if isinstance(v, int):
        return rng.choice([v + 1, v - 1, 0, -v, v * 2, v + 10, abs(v) + 3])
    if isinstance(v, float): return rng.choice([v + 1.0, v - 1.0, 0.0, -v, v * 2])
    if isinstance(v, str):
        c = [v[::-1], v + "a", v.upper() if v != v.upper() else v.lower(), "", v[1:], v + v]
        if v: c.append(v[:-1])
        return rng.choice(c)
    if isinstance(v, (list, tuple)):
        t = type(v); L = list(v); c = []
        c.append(L[::-1]); c.append([]); 
        if L:
            c.append(L[1:]); c.append(L[:-1]); c.append(L + [L[0]]); c.append(L + L)
            i = rng.randrange(len(L)); l2 = L[:]; l2[i] = mutate_value(L[i], rng); c.append(l2)
        r = rng.choice(c)
        return t(r) if t in (list, tuple) else r
    if isinstance(v, (set, frozenset)):
        L = sorted(v, key=repr)
        return type(v)(rng.choice([L[1:], L[:-1], L + [99], []]))
    if isinstance(v, dict):
        ks = list(v)
        if not ks: return {}
        d = dict(v); k = rng.choice(ks)
        if rng.random() < 0.5: del d[k]
        else: d[k] = mutate_value(d[k], rng)
        return d
    return v

def stable_rng(key):
    return random.Random(int(hashlib.sha256((key + str(SEED)).encode()).hexdigest()[:12], 16))

GEN_POOL = [0, 1, 5, -3, 12, "abc", "", "hello world", [1, 2, 3], [3, 1, 2, 1], [], [5, 5, 5], 2.5, (1, 2), {"a": 1, "b": 2}, "A1b2", [[1, 2], [3, 4]]]

def build_inputs(task_key, prompt, cand_codes):
    """inputs shared by all tiers of one task. returns (inputs, n_example, target, example_expected)"""
    exs = extract_examples(prompt)
    names = prompt_func_names(prompt)
    target = None
    for ex in exs:
        if ex["name"]: target = ("f", ex["name"]); break
    if target is None:
        for kind, nm, params in names:
            if kind == "m": target = ("m", nm); break
        if target is None and names: target = ("f", names[0][1])
    if target is None:
        # fall back to first def of any candidate
        for c in cand_codes:
            mm = re.search(r"^def\s+(\w+)", c, re.M)
            if mm: target = ("f", mm.group(1)); break
    params = []
    for kind, nm, ps in names:
        if target and (nm == target[1] or nm.endswith("." + target[1].split(".")[-1])): params = ps; break
    rng = stable_rng(task_key)
    inputs, exp = [], []
    base = [e for e in exs if (e["name"] in (None, (target or ("", ""))[1]))][:MAX_EX]
    for e in base:
        inputs.append(dict(args=e["args"], kw=e["kw"], src="example"))
        exp.append((e["has_exp"], e["exp"]))
    mutants = []
    if base:
        for _ in range(N_MUT * 3):
            e = base[rng.randrange(len(base))]
            args, kw = copy.deepcopy(e["args"]), copy.deepcopy(e["kw"])
            slots = [("a", i) for i in range(len(args))] + [("k", k) for k in kw]
            if not slots: continue
            kind, key = slots[rng.randrange(len(slots))]
            if kind == "a": args[key] = mutate_value(args[key], rng)
            else: kw[key] = mutate_value(kw[key], rng)
            m = dict(args=args, kw=kw, src="mutant")
            if m not in mutants and m not in inputs: mutants.append(m)
            if len(mutants) >= N_MUT: break
    else:
        npar = len(params) if params else 1
        for _ in range(N_MUT + 4):
            args = [copy.deepcopy(rng.choice(GEN_POOL)) for _ in range(npar)]
            m = dict(args=args, kw={}, src="fuzz")
            if m not in mutants: mutants.append(m)
    inputs += mutants
    return inputs, len(base), target, exp

# ----------------------------------------------------------------- sandbox exec
DRIVER = r'''
import sys, json, ast, signal, copy, io, os
src = open("cand.py").read()
spec = json.load(open("spec.json"))
inputs = [(ast.literal_eval(a["args"]), ast.literal_eval(a["kw"])) for a in spec["inputs"]]
def canon(v, d=0):
    if d > 6: return "<deep>"
    if isinstance(v, float):
        if v != v: return "nan"
        return round(v, 6)
    if isinstance(v, (set, frozenset)): return sorted((canon(x, d+1) for x in v), key=repr)
    if isinstance(v, (list, tuple)): return [canon(x, d+1) for x in v]
    if isinstance(v, dict): return sorted(([canon(k, d+1), canon(x, d+1)] for k, x in v.items()), key=repr)
    if isinstance(v, (int, str, bool)) or v is None: return v
    return "<%s>" % type(v).__name__
class TO(Exception): pass
def h(*a): raise TO()
signal.signal(signal.SIGALRM, h)
real = sys.stdout; sys.stdout = io.StringIO(); sys.stdin = io.StringIO("")
res = []
g = {"__name__": "cand_mod"}
try:
    signal.alarm(3); exec(compile(src, "cand.py", "exec"), g); signal.alarm(0)
    load_err = None
except BaseException as e:
    signal.alarm(0); load_err = type(e).__name__
tk, tn = spec["target"] if spec["target"] else (None, None)
fn = None
if load_err is None and tn:
    try:
        if tk == "m":
            cls, mt = tn.split(".")
            if cls not in g: cls = "Solution"
            inst = g[cls]()
            fn = getattr(inst, mt)
        else:
            fn = g.get(tn)
            if fn is None and "." in tn: fn = None
    except BaseException as e:
        fn = None
for a, kw in inputs:
    if load_err: res.append({"ok": False, "exc": "load:" + load_err}); continue
    if fn is None: res.append({"ok": False, "exc": "NoFunction"}); continue
    try:
        a2, k2 = copy.deepcopy(a), copy.deepcopy(kw)
        signal.alarm(2)
        r = fn(*a2, **k2)
        signal.alarm(0)
        c = canon(r)
        res.append({"ok": True, "out": json.dumps(c, default=str)[:300], "ty": type(r).__name__})
    except TO:
        res.append({"ok": False, "exc": "Timeout"})
    except BaseException as e:
        signal.alarm(0)
        res.append({"ok": False, "exc": type(e).__name__})
sys.stdout = real
print("@@RES@@" + json.dumps(res))
'''

def run_exec(code, inputs, target):
    from gwaya.sandbox import run_in_sandbox
    spec = {"inputs": [{"args": repr(i["args"]), "kw": repr(i["kw"])} for i in inputs], "target": list(target) if target else None}
    ok, out, err, to = run_in_sandbox(
        ["python3", "drv.py"], timeout_s=40, mem_mb=1024, cpu_s=35,
        files={"cand.py": code, "drv.py": DRIVER, "spec.json": json.dumps(spec)},
        env={"PYTHONHASHSEED": "0"})
    m = re.search(r"@@RES@@(.*)", out)
    if not m: return [{"ok": False, "exc": "DriverFail:" + ("timeout" if to else err[:60])} for _ in inputs]
    try: return json.loads(m.group(1))
    except Exception: return [{"ok": False, "exc": "DriverParse"} for _ in inputs]

# ----------------------------------------------------------------- stages
def canon_exp(v, d=0):
    if d > 6: return "<deep>"
    if isinstance(v, float):
        return "nan" if v != v else round(v, 6)
    if isinstance(v, (set, frozenset)): return sorted((canon_exp(x, d + 1) for x in v), key=repr)
    if isinstance(v, (list, tuple)): return [canon_exp(x, d + 1) for x in v]
    if isinstance(v, dict): return sorted(([canon_exp(k, d + 1), canon_exp(x, d + 1)] for k, x in v.items()), key=repr)
    if isinstance(v, (int, str, bool)) or v is None: return v
    return "<%s>" % type(v).__name__

def load_rows():
    rows = [json.loads(l) for l in open(DEV)]
    for r in rows:
        r["correct"] = int(r["correct"])
    return rows

def exec_stage(rows):
    by = C.defaultdict(dict)
    for r in rows: by[(r["set"], r["task_id"])][r["tier"]] = r
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    jobs = []
    task_meta = {}
    for key, tr in sorted(by.items()):
        any_r = next(iter(tr.values()))
        codes = {t: code_of(r["candidate"]) for t, r in tr.items()}
        inputs, nex, target, exp = build_inputs(key[0] + "|" + key[1], any_r["prompt"], list(codes.values()))
        task_meta["|".join(key)] = dict(n_inputs=len(inputs), n_example=nex, target=target, exp=exp,
                                        src=[i["src"] for i in inputs])
        for t, c in codes.items():
            ck = "|".join(key) + "|" + t
            if ck not in cache or cache[ck]["code_hash"] != hashlib.md5(c.encode()).hexdigest():
                jobs.append((ck, c, inputs, target))
    print("exec jobs", len(jobs), file=sys.stderr)
    def run(j):
        ck, c, inputs, target = j
        return ck, dict(code_hash=hashlib.md5(c.encode()).hexdigest(), res=run_exec(c, inputs, target))
    with ThreadPoolExecutor(6) as ex:
        for ck, v in ex.map(run, jobs):
            cache[ck] = v
    json.dump(cache, open(CACHE, "w"))
    return cache, task_meta, by

def static_feats(r, code):
    f = {}
    from gwaya.ast_audit import ZeroStubAudit
    f["trunc"] = int(r["finish_reason"] == "length")
    f["fence_open"] = int(r["candidate"].count("```") % 2 == 1)
    a = ZeroStubAudit.audit_python_code(code)
    f["stub_viol"] = len(a.violations)
    f["syntax_err"] = int(any("SyntaxError" in v for v in a.violations))
    f["nlines"] = math.log1p(len([l for l in code.split("\n") if l.strip()]))
    nd = 0; nn = 0; triv = 0; constret = 0; nret = 0
    try:
        t = ast.parse(code)
        for n in ast.walk(t):
            nn += 1
            if isinstance(n, ast.FunctionDef):
                nd += 1
                body = [b for b in n.body if not (isinstance(b, ast.Expr) and isinstance(getattr(b, "value", None), ast.Constant))]
                if len(body) <= 1: triv += 1
            if isinstance(n, ast.Return):
                nret += 1
                if n.value is None or isinstance(n.value, ast.Constant): constret += 1
    except SyntaxError:
        pass
    f["nnodes"] = math.log1p(nn); f["ndef"] = nd; f["trivial_def"] = triv
    f["const_ret_frac"] = constret / nret if nret else 1.0
    f["no_return"] = int(nret == 0)
    pcode = code_of(r["prompt"])
    f["sim_prompt"] = difflib.SequenceMatcher(None, pcode[:3000], code[:3000], autojunk=False).quick_ratio() if pcode else 0.0
    f["unchanged_prompt_code"] = int(pcode.strip() != "" and pcode.strip() == code.strip())
    f["mean_lp"] = r["mean_logprob"]; f["min_lp"] = r["min_logprob"]
    f["log_tok"] = math.log1p(r["completion_tokens"])
    f["n_vis"] = r["n_visible_tests"]
    return f

def hardcode_feat(code, exps):
    """candidate returns a literal that equals an example's expected value (non-trivial), or embeds an example arg"""
    try: t = ast.parse(code)
    except SyntaxError: return 0
    lits = []
    for n in ast.walk(t):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant):
            lits.append(n.value.value)
    hit = 0
    for he, e in exps:
        if he and isinstance(e, (int, str)) and not isinstance(e, bool):
            if (isinstance(e, str) and len(e) >= 2 or isinstance(e, int) and abs(e) >= 2) and e in lits: hit = 1
    return hit

def dyn_feats(key, tiers_res, tier, meta):
    f = {}
    res = tiers_res[tier]; n = len(res)
    src = meta["src"]; nex = meta["n_example"]
    exps = meta["exp"]
    # example pass fraction (prompt-visible examples with expected output)
    k_ok = k_n = 0
    for i in range(nex):
        he, e = exps[i]
        if not he: continue
        k_n += 1
        if res[i].get("ok") and res[i]["out"] == json.dumps(canon_exp(e), default=str)[:300]: k_ok += 1
    f["ex_n_exp"] = k_n
    f["ex_pass_frac"] = k_ok / k_n if k_n else 1.0
    f["ex_all_pass"] = int(k_n == 0 or k_ok == k_n)
    f["ex_fail_any"] = int(k_n > 0 and k_ok < k_n)
    f["has_examples"] = int(nex > 0)
    # informative inputs: at least one tier ok
    inf = [i for i in range(n) if any(tiers_res[t][i].get("ok") for t in tiers_res)]
    f["n_inf"] = len(inf)
    if inf:
        f["crash_rate"] = float(np.mean([not res[i].get("ok") for i in inf]))
        f["timeout_rate"] = float(np.mean([res[i].get("exc") == "Timeout" for i in inf]))
    else:
        f["crash_rate"] = 1.0; f["timeout_rate"] = 0.0
    mut = [i for i in inf if src[i] != "example"]
    f["crash_rate_mut"] = float(np.mean([not res[i].get("ok") for i in mut])) if mut else f["crash_rate"]
    f["no_function"] = int(any(r.get("exc") == "NoFunction" for r in res))
    f["load_err"] = int(any(str(r.get("exc", "")).startswith("load:") for r in res))
    f["bad_exc"] = float(np.mean([r.get("exc") in ("TypeError", "NameError", "AttributeError", "UnboundLocalError", "NoFunction") or str(r.get("exc", "")).startswith("load:")
                                 for r in res]))
    oks = [res[i]["out"] for i in range(n) if res[i].get("ok")]
    f["uniq_frac"] = len(set(oks)) / len(oks) if oks else 0.0
    f["ntypes"] = len({res[i]["ty"] for i in range(n) if res[i].get("ok")})
    other_uniq = max([len({tiers_res[t][i]["out"] for i in range(n) if tiers_res[t][i].get("ok")}) for t in tiers_res if t != tier] or [0])
    f["const_out"] = int(len(oks) >= 4 and len(set(oks)) <= 1 and other_uniq >= 3)
    # cross-tier agreement
    others = [t for t in tiers_res if t != tier]
    ag = []; agv = []
    for t in others:
        eq = []; eqv = []
        for i in range(n):
            a, b = res[i], tiers_res[t][i]
            if not a.get("ok") and not b.get("ok"): continue
            same = bool(a.get("ok") and b.get("ok") and a["out"] == b["out"])
            eq.append(same)
            if src[i] == "example": eqv.append(same)
        ag.append(float(np.mean(eq)) if eq else 0.0)
        agv.append(float(np.mean(eqv)) if eqv else float(np.mean(eq)) if eq else 0.0)
    f["agree_mean"] = float(np.mean(ag)) if ag else 0.0
    f["agree_min"] = float(np.min(ag)) if ag else 0.0
    f["agree_max"] = float(np.max(ag)) if ag else 0.0
    f["agree_vis_mean"] = float(np.mean(agv)) if agv else 0.0
    # plurality agreement
    pl = []
    for i in range(n):
        outs = [tiers_res[t][i]["out"] if tiers_res[t][i].get("ok") else None for t in tiers_res]
        c = C.Counter(o for o in outs if o is not None)
        if not c: continue
        top, cnt = c.most_common(1)[0]
        pl.append(1.0 if (res[i].get("ok") and res[i]["out"] == top and cnt >= 2) else 0.0)
    f["plural_agree"] = float(np.mean(pl)) if pl else 0.0
    return f

FS_STATIC = ["trunc", "fence_open", "stub_viol", "syntax_err", "nlines", "nnodes", "ndef", "trivial_def", "const_ret_frac",
             "no_return", "sim_prompt", "unchanged_prompt_code", "hardcode"]
FS_LP = ["mean_lp", "min_lp", "log_tok"]
FS_EX = ["ex_n_exp", "ex_pass_frac", "ex_all_pass", "ex_fail_any", "has_examples", "n_vis"]
FS_DIFF = ["n_inf", "crash_rate", "timeout_rate", "crash_rate_mut", "no_function", "load_err", "bad_exc", "uniq_frac", "ntypes", "const_out"]
FS_XT = ["agree_mean", "agree_min", "agree_max", "agree_vis_mean", "plural_agree"]
FS_LAYA = ["laya_p", "router_p"]

def build_features(rows, cache, task_meta, by):
    for key, tr in by.items():
        mk = "|".join(key); meta = task_meta[mk]
        tiers_res = {t: cache[mk + "|" + t]["res"] for t in tr}
        for t, r in tr.items():
            code = code_of(r["candidate"])
            f = static_feats(r, code)
            f["hardcode"] = hardcode_feat(code, [e for e in meta["exp"]])
            f.update(dyn_feats(key, tiers_res, t, meta))
            f["laya_p"] = float(r["laya_p"]) if r["laya_p"] not in (None, "") else 0.5
            f["router_p"] = float(r["router_p"]) if r["router_p"] not in (None, "") else 0.5
            r["_f"] = f

# ----------------------------------------------------------------- evaluation
def prec_stats(y, sel):
    sel = np.asarray(sel, bool); n = int(sel.sum()); k = int(np.asarray(y)[sel].sum())
    return dict(n=n, k=k, precision=(k / n if n else None), wilson=wilson(k, n), coverage=n / len(y))

def at_cov(y, score, cov=0.5):
    y = np.asarray(y); s = np.asarray(score, float)
    kk = int(math.ceil(cov * len(y)))
    o = np.argsort(-s, kind="stable")[:kk]
    m = np.zeros(len(y), bool); m[o] = True
    return prec_stats(y, m)

def curve(y, score):
    return {f"{c:.1f}": at_cov(y, score, c)["precision"] for c in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)}

def max_cov_at(y, score, target=0.9):
    """largest coverage whose top-k precision >= target (descriptive, selected on the same OOF scores: optimistic)"""
    y = np.asarray(y); o = np.argsort(-np.asarray(score, float), kind="stable")
    cs = np.cumsum(y[o]); best = 0
    for i in range(len(y)):
        if cs[i] / (i + 1) >= target and i + 1 >= 10: best = i + 1
    return best / len(y)

def oof_scores(rows, featnames, model="lr", n_splits=5):
    """OOF score for every row; trained on gate=verified rows only; failed rows = -1e9. GroupKFold by cluster."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
    X = np.array([[r["_f"][k] for k in featnames] for r in rows], float)
    y = np.array([r["correct"] for r in rows])
    g = np.array([r["set"] + "|" + r["cluster"] for r in rows])
    ver = np.array([r["gate"] == "verified" for r in rows])
    sc = np.full(len(rows), -1e9)
    idx = np.arange(len(rows))
    for tr, te in GroupKFold(n_splits).split(X, y, g):
        trv = tr[ver[tr]]
        if model == "lr":
            m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_depth=3, max_iter=80, learning_rate=0.08, random_state=0)
        if len(set(y[trv])) < 2:
            sc[te] = np.where(ver[te], 0.0, -1e9); continue
        m.fit(X[trv], y[trv])
        p = m.predict_proba(X[te])[:, 1]
        sc[te] = np.where(ver[te], p, -1e9)
    return sc

def auroc_verified(rows, sc):
    from sklearn.metrics import roc_auc_score
    ver = np.array([r["gate"] == "verified" for r in rows]); y = np.array([r["correct"] for r in rows])
    return float(roc_auc_score(y[ver], sc[ver])) if len(set(y[ver])) > 1 else None

def boot_ci(y, s1, s2, groups, B=1000):
    rng = np.random.RandomState(SEED)
    ug = sorted(set(groups)); gi = {u: np.where(np.asarray(groups) == u)[0] for u in ug}
    d1 = []; d2 = []; dd = []
    for _ in range(B):
        pick = rng.choice(len(ug), len(ug))
        ix = np.concatenate([gi[ug[i]] for i in pick])
        a = at_cov(y[ix], s1[ix])["precision"]; b = at_cov(y[ix], s2[ix])["precision"]
        d1.append(a); d2.append(b); dd.append(a - b)
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(first=q(d1), second=q(d2), diff=q(dd))

def main():
    rows = load_rows()
    cache, task_meta, by = exec_stage(rows)
    build_features(rows, cache, task_meta, by)
    sets = collections_sets = {
        "V0_gate_then_logprob": None,
        "V1_lp_lr": FS_LP,
        "V2_+static_audit": FS_LP + FS_STATIC,
        "V3_+prompt_examples": FS_LP + FS_STATIC + FS_EX,
        "V4_+differential_exec": FS_LP + FS_STATIC + FS_EX + FS_DIFF,
        "V5_+cross_tier_agreement": FS_LP + FS_STATIC + FS_EX + FS_DIFF + FS_XT,
        "V6_+laya_probs": FS_LP + FS_STATIC + FS_EX + FS_DIFF + FS_XT + FS_LAYA,
        "X_no_lp_all": FS_STATIC + FS_EX + FS_DIFF + FS_XT,
    }
    out = dict(config=dict(seed=SEED, n_mut=N_MUT, max_ex=MAX_EX, cv="GroupKFold(5) by set|cluster, trained on gate=verified rows",
                           model="LogisticRegression(C=0.3)+StandardScaler (primary a priori); HGB(depth3) as secondary"),
               feature_sets={k: v for k, v in sets.items() if v})
    # coverage of extraction
    out["extraction"] = {b: dict(tasks=sum(1 for k, tr in by.items() if next(iter(tr.values()))["benchmark"] == b),
                                 with_prompt_examples=sum(1 for k, tr in by.items() if next(iter(tr.values()))["benchmark"] == b and task_meta["|".join(k)]["n_example"] > 0))
                         for b in sorted({r["benchmark"] for r in rows})}
    out["tiers"] = {}
    for tier in TIERS:
        R = [r for r in rows if r["tier"] == tier]
        y = np.array([r["correct"] for r in R])
        ver = np.array([r["gate"] == "verified" for r in R])
        T = dict(n=len(R), accuracy=float(y.mean()), accuracy_wilson=wilson(int(y.sum()), len(y)),
                 gate_verified=prec_stats(y, ver),
                 dev_upper_bound_precision_at_cov50=min(1.0, float(y.mean()) / 0.5),
                 max_coverage_at_precision_0p9_oracle=float(min(1.0, y.mean() / 0.9)))
        lp = np.array([r["mean_logprob"] for r in R], float)
        base = np.where(ver, lp + 1e3, lp)   # gate-verified first, then logprob; failed ranked after
        variants = {}
        scores = {"V0_gate_then_logprob": base}
        for name, fs in sets.items():
            if fs is None: continue
            scores[name] = oof_scores(R, fs, "lr")
        scores["V5_HGB"] = oof_scores(R, sets["V5_+cross_tier_agreement"], "hgb")
        for name, sc in scores.items():
            variants[name] = dict(at_cov50=at_cov(y, sc, 0.5), curve=curve(y, sc), max_cov_at_prec_0p9_optimistic=max_cov_at(y, sc),
                                  auroc_within_gate_verified=auroc_verified(R, sc),
                                  by_set={s: at_cov(y[[r["set"] == s for r in R]], sc[[r["set"] == s for r in R]], 0.5)
                                          for s in ("F", "Eprime")},
                                  by_benchmark={b: at_cov(y[[r["benchmark"] == b for r in R]], sc[[r["benchmark"] == b for r in R]], 0.5)
                                                for b in sorted({r["benchmark"] for r in R})})
        T["variants"] = variants
        # rule-only add-ons applied to gate-verified answers (no fitting)
        F = [r["_f"] for r in R]
        rules = {
            "gate_verified": ver,
            "gate+no_stub": ver & np.array([f["stub_viol"] == 0 and not f["syntax_err"] for f in F]),
            "gate+not_truncated": ver & np.array([not f["trunc"] and not f["fence_open"] for f in F]),
            "gate+examples_all_pass": ver & np.array([f["ex_all_pass"] == 1 for f in F]),
            "gate+no_constant_output": ver & np.array([f["const_out"] == 0 for f in F]),
            "gate+crash_rate_mut<=0.5": ver & np.array([f["crash_rate_mut"] <= 0.5 for f in F]),
            "gate+agree_mean>=0.5": ver & np.array([f["agree_mean"] >= 0.5 for f in F]),
            "gate+agree_mean>=0.8": ver & np.array([f["agree_mean"] >= 0.8 for f in F]),
            "gate+all_rules(trunc,ex,const,crash,agree>=.5)": ver & np.array(
                [(not f["trunc"]) and f["stub_viol"] == 0 and f["ex_all_pass"] == 1 and f["const_out"] == 0
                 and f["crash_rate_mut"] <= 0.5 and f["agree_mean"] >= 0.5 for f in F]),
        }
        T["rule_addons"] = {k: prec_stats(y, v) for k, v in rules.items()}
        T["rule_addons_by_set"] = {s: {k: prec_stats(y[[r["set"] == s for r in R]], v[[r["set"] == s for r in R]]) for k, v in rules.items()}
                                   for s in ("F", "Eprime")}
        # paired cluster bootstrap: V5 vs V0 / V1 at cov 0.5
        groups = [r["set"] + "|" + r["cluster"] for r in R]
        T["bootstrap_V5_vs_V0"] = boot_ci(y, scores["V5_+cross_tier_agreement"], scores["V0_gate_then_logprob"], groups)
        T["bootstrap_V5_vs_V1_lp"] = boot_ci(y, scores["V5_+cross_tier_agreement"], scores["V1_lp_lr"], groups)
        # label-free feature sanity: univariate AUROC among gate-verified
        from sklearn.metrics import roc_auc_score
        ua = {}
        for k in FS_STATIC + FS_EX + FS_DIFF + FS_XT + FS_LP:
            v = np.array([f[k] for f in F], float)[ver]
            if len(set(v)) > 1 and len(set(y[ver])) > 1: ua[k] = float(roc_auc_score(y[ver], v))
        T["univariate_auroc_gate_verified"] = ua
        out["tiers"][tier] = T
    json.dump(out, open(OUT, "w"), indent=1, default=str)
    # concise print
    for tier, T in out["tiers"].items():
        print(f"== {tier} acc {T['accuracy']:.3f} gate_v prec {T['gate_verified']['precision']:.3f} cov {T['gate_verified']['coverage']:.3f}"
              f" | dev bound@cov.5 {T['dev_upper_bound_precision_at_cov50']:.3f}")
        for n, v in T["variants"].items():
            a = v["at_cov50"]
            print(f"  {n:28s} P@cov.5 {a['precision']:.3f} [{a['wilson'][0]:.3f},{a['wilson'][1]:.3f}]  maxcov@.9 {v['max_cov_at_prec_0p9_optimistic']:.2f}  auroc_v {v['auroc_within_gate_verified']}")
        for n, v in T["rule_addons"].items():
            print(f"   rule {n:50s} n {v['n']:3d} prec {v['precision'] if v['precision'] is None else round(v['precision'],3)} cov {v['coverage']:.3f}")

if __name__ == "__main__":
    main()
