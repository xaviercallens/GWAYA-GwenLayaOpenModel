#!/usr/bin/env python3
"""Dev-only baselines for A26. Never opens test_table.jsonl."""
import json, math, collections as C
import numpy as np
P="/mnt/data/home/xavkal/gwaya-data/trust_loop/dev_table.jsonl"
R=[json.loads(l) for l in open(P)]
def wil(k,n,z=1.96):
    if n==0: return (None,None)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); a=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-a)/d,(c+a)/d)
def prec(rows,mask):
    k=int(sum(r['correct'] for r,m in zip(rows,mask) if m)); n=int(sum(mask)); lo,hi=wil(k,n)
    return dict(n=n,k=k,precision=k/n if n else None,wilson=[lo,hi],coverage=n/len(rows))
def bal(y,pred):
    y=np.array(y);pred=np.array(pred)
    tpr=(pred[y==1]==1).mean() if (y==1).any() else float('nan')
    tnr=(pred[y==0]==0).mean() if (y==0).any() else float('nan')
    return dict(acc=float((pred==y).mean()),bal_acc=float((tpr+tnr)/2),tpr=float(tpr),tnr=float(tnr),n=len(y))
def at_cov(rows,score,cov=0.5):
    s=np.array([score(r) for r in rows],float); o=np.argsort(-s,kind='stable'); k=int(math.ceil(cov*len(rows)))
    m=np.zeros(len(rows),bool); m[o[:k]]=True; return prec(rows,m)
out={}
groups={'ALL':lambda r:True,'F':lambda r:r['set']=='F','Eprime':lambda r:r['set']=='Eprime'}
for tier in sorted({r['tier'] for r in R}):
  for g,f in groups.items():
    rows=[r for r in R if r['tier']==tier and f(r)]
    if not rows: continue
    y=[r['correct'] for r in rows]; k=sum(y)
    e={'n':len(rows),'accuracy':k/len(rows),'wilson':wil(k,len(rows))}
    e['U_always_accept']=bal(y,[1]*len(rows))
    e['D_gate_verified']=prec(rows,[r['gate']=='verified' for r in rows])
    e['D_gate_counts']=dict(C.Counter(r['gate'] for r in rows))
    e['D_meanlogprob_cov50']=at_cov(rows,lambda r:r['mean_logprob'])
    e['D_minlogprob_cov50']=at_cov(rows,lambda r:r['min_logprob'])
    gv=[r for r in rows if r['gate']=='verified']
    e['D_gate_verified_then_lp_cov50']=at_cov(gv,lambda r:r['mean_logprob']) if gv else None
    for name in('router_p','laya_p'):
      sub=[r for r in rows if r[name] is not None]
      if sub:
        yy=[r['correct'] for r in sub]
        e[f'U_{name}>=0.5']=bal(yy,[int(r[name]>=0.5) for r in sub])
        e[f'D_{name}_cov50']=at_cov(sub,lambda r:r[name])
        from sklearn.metrics import roc_auc_score
        e[f'{name}_auroc']=float(roc_auc_score(yy,[r[name] for r in sub]))
    out[f'{tier}|{g}']=e
json.dump(out,open('/mnt/data/home/xavkal/gwaya-data/trust_loop/baseline_dev.json','w'),indent=1,default=str)
for k,e in out.items():
    print(k,'n',e['n'],'acc %.3f'%e['accuracy'],
      'gate_v prec %.3f cov %.2f'%(e['D_gate_verified']['precision'] or -1,e['D_gate_verified']['coverage']),
      'lp50 %.3f'%e['D_meanlogprob_cov50']['precision'],
      'router>=.5 acc/bal %s'%({k2:round(v,3) for k2,v in e.get('U_router_p>=0.5',{}).items() if k2 in('acc','bal_acc')}),
      'laya>=.5 %s'%({k2:round(v,3) for k2,v in e.get('U_laya_p>=0.5',{}).items() if k2 in('acc','bal_acc')}))
