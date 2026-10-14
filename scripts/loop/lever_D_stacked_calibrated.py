#!/usr/bin/env python3
"""Lever D3: stacked calibrated downstream scorer (A26). Reads dev_table only; seed 0.
Grouped CV by cluster; calibration + threshold chosen inside each outer training fold."""
import json, math, ast, re, hashlib, collections as C, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
SRC="/mnt/data/home/xavkal/gwaya-data/trust_loop/dev_table.jsonl"
OUT="/mnt/data/home/xavkal/gwaya-data/trust_loop/lever_D_stacked_calibrated.json"
R=[json.loads(l) for l in open(SRC)]
def wil(k,n,z=1.96):
    if n==0: return [None,None]
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); a=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return [(c-a)/d,(c+a)/d]
def norm(c):
    c=re.sub(r"```(?:python)?","",c or ""); return re.sub(r"\s+","",re.sub(r"#.*","",c))
# cross-tier agreement (label free, visible text only)
byt=C.defaultdict(dict)
for r in R: byt[(r['set'],r['task_id'])][r['tier']]=norm(r['candidate'])
def feats(r):
    c=r['candidate'] or ""; code=re.sub(r"```(?:python)?","",c)
    try: t=ast.parse(code); ok=1; nd=sum(1 for _ in ast.walk(t))
    except Exception: ok=0; nd=0
    stub=int(bool(re.search(r"NotImplemented|^\s*pass\s*$|\.\.\.\s*$",code,re.M)))
    oth=[v for t,v in byt[(r['set'],r['task_id'])].items() if t!=r['tier']]
    agree=sum(v==norm(c) for v in oth)/len(oth) if oth else 0.0
    return dict(gate=int(r['gate']=='verified'),mlp=r['mean_logprob'],minlp=max(r['min_logprob'],-20),
      logtok=math.log1p(r['completion_tokens']),trunc=int(r['finish_reason']=='length'),
      laya=math.log(max(r['laya_p'],1e-4)/(1-min(r['laya_p'],1-1e-4))),nvis=r['n_visible_tests'],
      parse=ok,nodes=math.log1p(nd),stub=stub,agree=agree,plen=math.log1p(len(r['prompt'])),
      tier2=int('2b' in r['tier']),tier4=int('4b' in r['tier']),
      bcb=int(r['benchmark']=='BCB'),fsb=int(r['benchmark']=='FSB'),lc=int(r['benchmark']=='LeetCode'))
F=[feats(r) for r in R]
y=np.array([r['correct'] for r in R]); clus=np.array([r['cluster'] for r in R])
SETS={'core':['gate','mlp','minlp','logtok','trunc','nvis'],
      'core+laya':['gate','mlp','minlp','logtok','trunc','nvis','laya'],
      'core+ast+agree':['gate','mlp','minlp','logtok','trunc','nvis','parse','nodes','stub','agree','tier2','tier4'],
      'full':list(F[0].keys())}
def X(cols): return np.array([[f[c] for c in cols] for f in F],float)
def fold_of(c,k): return int(hashlib.md5(("0|"+c).encode()).hexdigest(),16)%k
def mk(kind):
    if kind=='lr': return make_pipeline(StandardScaler(),LogisticRegression(C=0.3,max_iter=2000))
    return HistGradientBoostingClassifier(max_depth=2,max_iter=80,learning_rate=0.05,min_samples_leaf=20,random_state=0)
def oof(kind,Xm,idx,k=5):
    """inner OOF raw scores over idx, grouped by cluster"""
    s=np.zeros(len(idx)); f=np.array([fold_of(clus[i],k) for i in idx])
    for j in range(k):
        tr=idx[f!=j]; te=f==j
        if te.sum()==0: continue
        m=mk(kind).fit(Xm[tr],y[tr]); s[te]=m.predict_proba(Xm[idx[te]])[:,1]
    return s
def pick_thr(sc,yy,target=0.90,mincov=0.5):
    o=np.argsort(-sc,kind='stable'); ys=yy[o]; cs=np.cumsum(ys); n=len(ys)
    best=None
    for i in range(int(math.ceil(0.05*n)),n+1):  # largest coverage with Wilson LB>=target
        if wil(cs[i-1],i)[0]>=target: best=i
    if best is None:  # fall back: largest coverage with point precision>=target, else mincov
        for i in range(int(math.ceil(0.05*n)),n+1):
            if cs[i-1]/i>=target: best=i
    if best is None: best=int(math.ceil(mincov*n))
    return sc[o[best-1]],best/n
TARGETS=[0.80,0.85,0.90]
def run(kind,cols,K=5):
    Xm=X(cols); n=len(y); fo=np.array([fold_of(c,K) for c in clus])
    sc=np.zeros(n); acc={t:np.zeros(n,bool) for t in TARGETS}; thrs={t:[] for t in TARGETS}
    for j in range(K):
        tr=np.where(fo!=j)[0]; te=np.where(fo==j)[0]
        raw=oof(kind,Xm,tr)
        iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(raw,y[tr])
        cal=iso.predict(raw)+1e-6*raw
        m=mk(kind).fit(Xm[tr],y[tr]); rte=m.predict_proba(Xm[te])[:,1]
        sc[te]=iso.predict(rte)+1e-6*rte
        for t in TARGETS:
            thr,_=pick_thr(cal,y[tr],target=t); thrs[t].append(float(thr)); acc[t][te]=sc[te]>=thr
    return sc,acc,thrs
def summ(mask,yy):
    k=int(yy[mask].sum()); nn=int(mask.sum())
    return dict(n_accept=nn,coverage=nn/len(yy),precision=k/nn if nn else None,wilson=wil(k,nn))
def curve(sc,yy):
    o=np.argsort(-sc,kind='stable'); out={}
    for cov in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        m=np.zeros(len(yy),bool); m[o[:int(math.ceil(cov*len(yy)))]]=True; out[str(cov)]=summ(m,yy)
    return out
tiers=np.array([r['tier'] for r in R]); sets=np.array([r['set'] for r in R]); bench=np.array([r['benchmark'] for r in R])
def subs(sc,accd):
    d={}
    for name,m in [('ALL',np.ones(len(y),bool))]+[(t,tiers==t) for t in sorted(set(tiers))]+[(s,sets==s) for s in sorted(set(sets))]+[(b,bench==b) for b in sorted(set(bench))]:
        d[name]=dict(base_rate=float(y[m].mean()),n=int(m.sum()),auroc=float(roc_auc_score(y[m],sc[m])),
            at_cv_threshold={str(t):summ(accd[t][m],y[m]) for t in TARGETS},curve=curve(sc[m],y[m]))
    return d
res={}
for kind in ['lr','gbm']:
    for sn,cols in SETS.items():
        sc,acc,thrs=run(kind,cols); res[f'{kind}|{sn}']=dict(thresholds={str(t):v for t,v in thrs.items()},**subs(sc,acc))
# hand rule baselines (no fitting): gate verified; gate verified then top-50% by mean logprob within slice
def hand(mask):
    g=np.array([r['gate']=='verified' for r in R]); mlp=np.array([r['mean_logprob'] for r in R])
    out={'gate_verified':summ(g[mask],y[mask])}
    idx=np.where(mask&g)[0]; o=idx[np.argsort(-mlp[idx],kind='stable')]
    d={}
    for cov in [0.3,0.4,0.5]:
        k=int(math.ceil(cov*mask.sum())); sel=o[:k]
        d[str(cov)]=dict(n_accept=len(sel),coverage=len(sel)/mask.sum(),precision=float(y[sel].mean()) if len(sel) else None,
          wilson=wil(int(y[sel].sum()),len(sel)))
    out['gate_then_logprob']=d
    out['logprob_only_cov50']=curve(mlp[mask],y[mask])['0.5']
    return out
res['hand_rule']={'ALL':hand(np.ones(len(y),bool)),**{t:hand(tiers==t) for t in sorted(set(tiers))}}
res['meta']=dict(n=len(y),clusters=len(set(clus)),folds=5,seed=0,note="laya_p for F rows comes from laya_v1_calibrator_F; if that calibrator saw F labels, 'laya' features are optimistic on F. 'core' and 'core+ast+agree' exclude laya_p.")
json.dump(res,open(OUT,'w'),indent=1)
for k,v in res.items():
    if k in('hand_rule','meta'):continue
    for s_ in ['ALL','qwen3.5-4b-bf16','qwen3.5-2b-bf16','qwen3.5-9b-bf16']:
        c=v[s_]['curve']['0.5']; out=[]
        for t in ["0.8","0.85","0.9"]:
            a=v[s_]['at_cv_threshold'][t]
            out.append("t%s cov %.2f p %s [%s]"%(t,a['coverage'],None if a['precision'] is None else round(a['precision'],3),None if a['precision'] is None else "%.3f,%.3f"%tuple(a['wilson'])))
        print("%-18s %-14s %s | p@cov50 %.3f [%.3f,%.3f] auroc %.3f"%(k,s_[-12:],' | '.join(out),c['precision'],*c['wilson'],v[s_]['auroc']))
for s_,v in res['hand_rule'].items():
    print('HAND',s_,'gate cov',round(v['gate_verified']['coverage'],2),'p',round(v['gate_verified']['precision'],3),'gate+lp50',v['gate_then_logprob']['0.5'],'lp50',round(v['logprob_only_cov50']['precision'],3))
