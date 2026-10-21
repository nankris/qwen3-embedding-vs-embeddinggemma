"""Supplemental BEIR/trec_eval-compatible linear-gain nDCG; never changes primary verdict."""
from pathlib import Path
import json,math
import numpy as np,pandas as pd
R=Path('results');DS=['scifact','nfcorpus','arguana'];KS=['qwen','gemma','bge'];rng=np.random.default_rng(98);boots={};rows=[];raw=[]
for ds in DS:
 d=json.load(open(R/(ds+'_data.json')));rel={}
 for r in d['qrels']:
  if r['score']>0:rel.setdefault(r['qid'],{})[r['did']]=r['score']
 ids=sorted(rel);ix=rng.integers(len(ids),size=(10000,len(ids)))
 for k in KS:
  run=json.load(open(R/(k+'_'+ds+'_top100.json')));scores=[]
  for qid in ids:
   rr=rel[qid];rank=run[qid][:10];dcg=sum(rr.get(x,0)/math.log2(i+2) for i,x in enumerate(rank));ideal=sum(v/math.log2(i+2) for i,v in enumerate(sorted(rr.values(),reverse=True)[:10]));v=dcg/ideal;scores.append(v);raw.append({'model':k,'dataset':ds,'qid':qid,'ndcg10_linear':v})
  a=np.array(scores);bs=a[ix].mean(1);boots[k,ds]=bs;rows.append({'model':k,'dataset':ds,'mean':a.mean(),'lo':np.percentile(bs,2.5),'hi':np.percentile(bs,97.5)})
for k in KS:
 bs=np.mean([boots[k,d] for d in DS],0);boots[k,'macro']=bs;mu=np.mean([next(r['mean'] for r in rows if r['model']==k and r['dataset']==d) for d in DS]);rows.append({'model':k,'dataset':'macro','mean':mu,'lo':np.percentile(bs,2.5),'hi':np.percentile(bs,97.5)})
diff=boots['qwen','macro']-boots['gemma','macro'];point=next(x['mean'] for x in rows if x['model']=='qwen' and x['dataset']=='macro')-next(x['mean'] for x in rows if x['model']=='gemma' and x['dataset']=='macro')
(R/'linear_gain_paired_macro.json').write_text(json.dumps({'difference':point,'lo':np.percentile(diff,2.5),'hi':np.percentile(diff,97.5),'note':'Qwen minus Google; supplemental sensitivity only'},indent=2));pd.DataFrame(raw).to_csv(R/'linear_gain_per_query.csv',index=False);pd.DataFrame(rows).to_csv(R/'linear_gain_summary.csv',index=False)
