import json
from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('results');F=Path('figures');KS=['qwen','gemma','bge'];NAMES={'qwen':'Qwen3 0.6B','gemma':'EmbeddingGemma 2','bge':'BGE small'}
colors=['#3568a8','#e58b2c','#27866e']
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
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
