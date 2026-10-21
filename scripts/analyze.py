"""Recompute query-paired stratified bootstrap, tables, and publication PNGs."""
import json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('results');F=Path('figures');F.mkdir(exist_ok=True)
DS=['scifact','nfcorpus','arguana'];KS=['qwen','gemma','bge'];MET=['ndcg10','recall10','recall100'];NAMES={'qwen':'Qwen3 0.6B','gemma':'EmbeddingGemma 2','bge':'BGE small'}
B=10000;rng=np.random.default_rng(98);summ=[];boot={};means={}
for ds in DS+['miracl_hi','miracl_te']:
 frames={k:pd.read_csv(R/(k+'_'+ds+'_metrics.csv')).sort_values('qid') for k in KS if (R/(k+'_'+ds+'_metrics.csv')).exists()}
 if not frames:continue
 assert all(list(f.qid)==list(next(iter(frames.values())).qid) for f in frames.values()), 'Query mismatch'
 n=len(next(iter(frames.values())));ix=rng.integers(n,size=(B,n))
 for k,df in frames.items():
  arr=df[MET].to_numpy();mu=arr.mean(0);bs=arr[ix].mean(1);boot[k,ds]=bs;means[k,ds]=mu
  for j,m in enumerate(MET):summ.append({'model':k,'dataset':ds,'metric':m,'mean':mu[j],'lo':np.percentile(bs[:,j],2.5),'hi':np.percentile(bs[:,j],97.5),'queries':n})
for k in KS:
 if not all((k,ds) in boot for ds in DS):continue
 bs=np.mean([boot[k,ds] for ds in DS],0);mu=np.mean([means[k,ds] for ds in DS],0);boot[k,'macro']=bs;means[k,'macro']=mu
 for j,m in enumerate(MET):summ.append({'model':k,'dataset':'macro','metric':m,'mean':mu[j],'lo':np.percentile(bs[:,j],2.5),'hi':np.percentile(bs[:,j],97.5),'queries':sum(next(x['queries'] for x in summ if x['model']==k and x['dataset']==ds) for ds in DS)})
s=pd.DataFrame(summ);s.to_csv(R/'summary.csv',index=False)
pairs=[]
for a,b in [('qwen','gemma'),('qwen','bge'),('gemma','bge')]:
 for ds in DS+['macro','miracl_hi','miracl_te']:
  if (a,ds) not in boot or (b,ds) not in boot:continue
  delta=boot[a,ds]-boot[b,ds];mu=means[a,ds]-means[b,ds]
  for j,m in enumerate(MET):pairs.append({'a':a,'b':b,'dataset':ds,'metric':m,'difference':mu[j],'lo':np.percentile(delta[:,j],2.5),'hi':np.percentile(delta[:,j],97.5)})
pd.DataFrame(pairs).to_csv(R/'paired_differences.csv',index=False)
if ('qwen','macro') in boot and ('gemma','macro') in boot:
 d=next(x for x in pairs if x['a']=='qwen' and x['b']=='gemma' and x['dataset']=='macro' and x['metric']=='ndcg10');quality='PROVEN' if d['lo']>0 else 'DISPROVEN' if d['hi']<0 else 'INCONCLUSIVE'
 win='qwen' if means['qwen','macro'][1]>means['gemma','macro'][1] else 'gemma';lose='gemma' if win=='qwen' else 'qwen';rat=100*(boot[win,'macro'][:,1]/boot[lose,'macro'][:,1]-1)
 headline={'winner':NAMES[win],'relative_recall10_pct':100*(means[win,'macro'][1]/means[lose,'macro'][1]-1),'lo':np.percentile(rat,2.5),'hi':np.percentile(rat,97.5),'quality_verdict':quality}
 (R/'headline.json').write_text(json.dumps(headline,indent=2))
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
colors=['#3568a8','#e58b2c','#27866e']
fig,ax=plt.subplots(figsize=(10,5))
for i,k in enumerate(KS):
 z=s[(s.model==k)&s.dataset.isin(DS)&(s.metric=='ndcg10')].set_index('dataset').reindex(DS)
 if z['mean'].isna().all():continue
 x=np.arange(3)+(i-1)*.23;ax.bar(x,z['mean']*100,width=.22,label=NAMES[k],color=colors[i]);ax.errorbar(x,z['mean']*100,yerr=np.array([z['mean']-z.lo,z.hi-z['mean']])*100,fmt='none',color='#222',capsize=3)
ax.set_xticks(range(3),['SciFact','NFCorpus','ArguAna']);ax.set_ylabel('nDCG@10 · exponential gains (×100; 95% CI)');ax.set_title('English retrieval: full document collections');ax.legend(fontsize=10);fig.tight_layout();fig.savefig(F/'quality.png');plt.close(fig)
speeds=[]
for k in KS:
 row={'model':NAMES[k]}
 for dev in ['gpu','cpu']:
  p=R/(k+'_'+dev+'_speed.json');row[dev]=json.loads(p.read_text())['docs_per_second'] if p.exists() else np.nan
 speeds.append(row)
pd.DataFrame(speeds).to_csv(R/'speeds.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for ax,dev,title in zip(axes,['gpu','cpu'],['NVIDIA L4 · bfloat16 · batch 16','Sandbox ARM CPU · float32 · batch 1']):
 vals=[x[dev] for x in speeds];ax.bar(range(3),vals,color=colors);ax.set_xticks(range(3),[NAMES[k] for k in KS],rotation=15,ha='right',fontsize=10);ax.set_ylabel('Documents / second');ax.set_title(title,fontsize=12)
 for i,v in enumerate(vals):
  if np.isfinite(v):ax.text(i,v,f'{v:.1f}',ha='center',va='bottom')
fig.suptitle('Warm encoding throughput · same 500 SciFact documents',fontsize=14);fig.tight_layout();fig.savefig(F/'speed.png');plt.close(fig)
rows=[]
for ds in DS+['macro']:
 for k in KS:
  z=s[(s.model==k)&(s.dataset==ds)].set_index('metric')
  if z.empty:continue
  rows.append([ds if ds!='macro' else 'Equal-dataset mean',NAMES[k]]+[f"{100*z.loc[m,'mean']:.1f} [{100*z.loc[m,'lo']:.1f}, {100*z.loc[m,'hi']:.1f}]" for m in MET])
fig,ax=plt.subplots(figsize=(13,7));ax.axis('off');tab=ax.table(cellText=rows,colLabels=['Dataset','Model','nDCG@10 (exp.)','Recall@10','Recall@100'],loc='center',cellLoc='center',colWidths=[.19,.21,.20,.20,.20]);tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,1.8)
for (r,c),cell in tab.get_celld().items():
 cell.set_edgecolor('#ddd')
 if r==0:cell.set_facecolor('#193b58');cell.set_text_props(color='white',weight='bold')
 elif r%2:cell.set_facecolor('#eef3f7')
ax.set_title('English retrieval · scores ×100 [95% bootstrap CI]',fontsize=16,pad=15);fig.tight_layout();fig.savefig(F/'table.png');plt.close(fig)
mini=s[s.dataset.isin(['miracl_hi','miracl_te']) & (s.metric=='recall10')]
if len(mini):
 fig,ax=plt.subplots(figsize=(9,5))
 for i,k in enumerate(KS):
  z=mini[mini.model==k].set_index('dataset').reindex(['miracl_hi','miracl_te']);x=np.arange(2)+(i-1)*.24;ax.bar(x,z['mean']*100,.23,color=colors[i],label=NAMES[k]);ax.errorbar(x,z['mean']*100,yerr=np.array([z['mean']-z.lo,z.hi-z['mean']])*100,fmt='none',color='#222',capsize=3)
 ax.set_xticks(range(2),['Hindi','Telugu']);ax.set_ylabel('Recall@10 (×100; 95% CI)');ax.set_title('Exploratory MIRACL check · 200 queries/language\nPooled positives + 1,000 distractors; NOT full-corpus retrieval');ax.legend(fontsize=10);fig.tight_layout();fig.savefig(F/'multilingual.png');plt.close(fig)
print(s.to_string(index=False))
