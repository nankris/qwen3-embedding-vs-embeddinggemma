"""Exploratory pooled-candidate check, NOT full MIRACL benchmark."""
import run_eval as E
import gzip,requests,json,time
from huggingface_hub import hf_hub_download,HfApi
import numpy as np,pandas as pd,torch
from pathlib import Path
R=E.R
def prep(lang):
 p=R/('miracl_'+lang+'_data.json')
 if p.exists():return json.loads(p.read_text())
 repo='miracl/miracl';rev=HfApi().dataset_info(repo).sha
 base='miracl-v1.0-'+lang
 qf=hf_hub_download(repo,base+'/topics/topics.'+base+'-dev.tsv',repo_type='dataset',revision=rev)
 q=dict(line.rstrip('\n').split('\t',1) for line in open(qf));ids=sorted(q)[:200];q={i:q[i] for i in ids}
 rf=hf_hub_download(repo,base+'/qrels/qrels.'+base+'-dev.tsv',repo_type='dataset',revision=rev)
 rel=[];wanted=set()
 for line in open(rf):
  qid,_,did,score=line.split()
  if qid in q and int(score)>0:rel.append({'qid':qid,'did':did,'score':int(score)});wanted.add(did)
 cr='miracl/miracl-corpus';rv=HfApi().dataset_info(cr).sha;corpus={};distractors=0
 for f in HfApi().list_repo_files(cr,repo_type='dataset',revision=rv):
  if not f.startswith('miracl-corpus-v1.0-'+lang+'/') or not f.endswith('.gz'):continue
  path=hf_hub_download(cr,f,repo_type='dataset',revision=rv)
  with gzip.open(path,'rt') as stream:
   for line in stream:
    x=json.loads(line);did=x.get('docid',x.get('_id'))
    if did in wanted or distractors<1000:
     corpus[did]={'id':did,'title':x.get('title',''),'text':x['text']}
     if did not in wanted:distractors+=1
 d={'queries':q,'qrels':rel,'corpus':sorted(corpus.values(),key=lambda x:x['id']),'revisions':{repo:rev,cr:rv},'protocol':'first 200 lexicographic dev query IDs, all their positive documents plus first 1000 nonpositive corpus documents; easy pooled candidates, not full retrieval'}
 assert wanted<=set(corpus)
 E.save(p,d);return d
if __name__=='__main__':
 torch.set_num_threads(4)
 data={l:prep(l) for l in ['hi','te']}
 for k,repo in E.MODELS.items():
  if all((R/(k+'_miracl_'+l+'_metrics.csv')).exists() for l in data):continue
  try:
   m=E.SentenceTransformer(repo,revision=json.load(open(R/(k+'_model.json')))['revision'],device='cuda',model_kwargs={'torch_dtype':torch.bfloat16},config_kwargs={'vision_config':None,'audio_config':None} if k=='gemma' else {});m.max_seq_length=512
   for lang,d in data.items():
    out=R/(k+'_miracl_'+lang+'_metrics.csv')
    if out.exists():continue
    de=E.enc(m,k,E.texts(k,d['corpus']));ids=sorted(d['queries']);qe=E.enc(m,k,[d['queries'][i] for i in ids],True);assert np.isfinite(de).all() and np.isfinite(qe).all()
    rel={}
    for r in d['qrels']:rel.setdefault(r['qid'],{})[r['did']]=r['score']
    rows=[];runs={};docids=[x['id'] for x in d['corpus']]
    for qid,s in zip(ids,qe@de.T):
     rank=[docids[i] for i in np.argsort(-s,kind='stable')[:100]];rr=rel[qid];runs[qid]=rank;hits=[int(x in rr) for x in rank]
     ideal=sum(1/np.log2(i+2) for i in range(min(10,len(rr))));ndcg=sum(h/np.log2(i+2) for i,h in enumerate(hits[:10]))/ideal
     rows.append({'qid':qid,'ndcg10':ndcg,'recall10':sum(hits[:10])/len(rr),'recall100':sum(hits)/len(rr)})
    E.save(R/(k+'_miracl_'+lang+'_top100.json'),runs);E.csvsave(pd.DataFrame(rows),out);print('MINI',k,lang,len(rows),len(docids),flush=True)
   del m;E.gc.collect();torch.cuda.empty_cache()
  except Exception:E.save(R/(k+'_mini_error.json'),{'error':E.traceback.format_exc()})
