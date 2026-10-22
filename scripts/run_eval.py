import os,json,time,sys,gc,subprocess,traceback
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
import numpy as np,torch,pandas as pd
from huggingface_hub import HfApi,hf_hub_download
from sentence_transformers import SentenceTransformer
ROOT=Path(os.environ.get('EXP_ROOT','/content/embedding-comparison')); ROOT.mkdir(exist_ok=True,parents=True)
R=ROOT/'results'; R.mkdir(exist_ok=True)
MODELS={'qwen':'Qwen/Qwen3-Embedding-0.6B','gemma':'google/embeddinggemma-2','bge':'BAAI/bge-small-en-v1.5'}
def save(p,d):
 t=Path(str(p)+'.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def csvsave(df,p):
 t=Path(str(p)+'.tmp');df.to_csv(t,index=False);t.replace(p)
def data(ds):
 p=R/(ds+'_data.json')
 if p.exists():return json.loads(p.read_text())
 repo='BEIR/'+ds; rev=HfApi().dataset_info(repo).sha
 def frame(config):return pd.read_parquet(hf_hub_download(repo,config+'/'+config+'-00000-of-00001.parquet',repo_type='dataset',revision=rev))
 c=frame('corpus');q=frame('queries')
 qr='BEIR/'+ds+'-qrels';rv=HfApi().dataset_info(qr).sha
 files=HfApi().list_repo_files(qr,repo_type='dataset',revision=rv)
 f=next(f for f in files if f.endswith('.tsv') and ('test' in f))
 rel=pd.read_csv(hf_hub_download(qr,f,repo_type='dataset',revision=rv),sep='\t')
 rel.columns=['qid','did','score']; rel['qid']=rel.qid.astype(str);rel['did']=rel.did.astype(str)
 d={'corpus':sorted([{'id':str(x['_id']),'title':str(x.get('title','') or ''),'text':x['text']} for x in c.to_dict('records')],key=lambda x:x['id']), 'queries':{str(x['_id']):x['text'] for x in q.to_dict('records')},'qrels':rel.to_dict('records'),'revisions':{repo:rev,qr:rv}}
 save(p,d);return d
def texts(k,d):
 if k=='gemma':return ['title: '+(x['title'] or 'none')+' | text: '+x['text'] for x in d]
 return [(x['title']+' '+x['text']).strip() for x in d]
def enc(m,k,ss,query=False,bs=16):
 kw={'prompt':''}
 if query:kw={'prompt':{'qwen':'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ','gemma':'task: search result | query: ','bge':'Represent this sentence for searching relevant passages: '}[k]}
 return m.encode(ss,batch_size=bs,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False,**kw)
def main():
 cpu='--cpu' in sys.argv; device='cpu' if cpu else 'cuda';torch.set_num_threads(2 if cpu else 4)
 save(R/('cpu_environment.json' if cpu else 'gpu_environment.json'),{'torch':torch.__version__,'platform':subprocess.getoutput('uname -a'),'gpu':subprocess.getoutput('nvidia-smi'),'packages':subprocess.getoutput(sys.executable+' -m pip freeze')})
 datasets={ds:data(ds) for ds in ['scifact','nfcorpus','arguana']}
 for k,repo in MODELS.items():
  speed=R/(k+('_cpu_speed.json' if cpu else '_gpu_speed.json'))
  if cpu and speed.exists():continue
  if not cpu and all((R/(k+'_'+ds+'_metrics.csv')).exists() for ds in datasets) and speed.exists():continue
  try:
   rev=json.loads((R/(k+'_model.json')).read_text())['revision'] if (R/(k+'_model.json')).exists() else HfApi().model_info(repo).sha;t=time.perf_counter()
   m=SentenceTransformer(repo,revision=rev,device=device,model_kwargs={'torch_dtype':torch.float32,'disable_mmap':True} if cpu else {'torch_dtype':torch.bfloat16},config_kwargs={'vision_config':None,'audio_config':None} if k=='gemma' else {})
   m.max_seq_length=512
   print('LOADED',k,'seconds',time.perf_counter()-t,flush=True)
   save(R/(k+('_cpu_model.json' if cpu else '_model.json')),{'repo':repo,'revision':rev,'parameters':sum(p.numel() for p in m.parameters()),'embedding_dim':m.get_sentence_embedding_dimension(),'max_seq_length':m.max_seq_length,'prompts':m.prompts,'architecture':str(m),'transformers_version':sys.modules['transformers'].__version__,'sentence_transformers_version':sys.modules['sentence_transformers'].__version__})
   sample=texts(k,datasets['scifact']['corpus'][:500]);bs=1 if cpu else 16
   if not speed.exists():
    enc(m,k,sample[:16],bs=bs)
    if not cpu:torch.cuda.synchronize()
    t=time.perf_counter();a=enc(m,k,sample,bs=bs)
    if not cpu:torch.cuda.synchronize()
    elapsed=time.perf_counter()-t
    assert np.isfinite(a).all()
    save(speed,{'docs':500,'seconds':elapsed,'docs_per_second':500/elapsed,'batch_size':bs,'dtype':'float32' if cpu else 'bfloat16','device':device})
    print('SPEED',k,device,500/elapsed,'docs/s',elapsed,'seconds',flush=True)
   if not cpu:
    for ds,d in datasets.items():
     out=R/(k+'_'+ds+'_metrics.csv')
     if out.exists():continue
     t=time.perf_counter();cp=R/(k+'_'+ds+'_docs.npy')
     if cp.exists():
      try:de=np.load(cp)
      except Exception:cp.unlink();de=None
     else:de=None
     if de is None:
      de=enc(m,k,texts(k,d['corpus']));assert np.isfinite(de).all();np.save(str(cp)+'.tmp.npy',de);Path(str(cp)+'.tmp.npy').replace(cp)
     rel={}
     for r in d['qrels']:
      if r['score']>0:rel.setdefault(r['qid'],{})[r['did']]=r['score']
     ids=sorted(rel);qe=enc(m,k,[d['queries'][i] for i in ids],True);assert np.isfinite(qe).all()
     docids=[x['id'] for x in d['corpus']];rows=[];runs={}
     for start in range(0,len(ids),64):
      sim=qe[start:start+64]@de.T
      for j,s in enumerate(sim):
       qid=ids[start+j]
       if qid in docids:s[docids.index(qid)]=-np.inf
       ix=np.argsort(-s,kind='stable')[:100];rank=[docids[i] for i in ix];rr=rel[qid];hits=[int(x in rr) for x in rank]
       dcg=sum((2**rr.get(x,0)-1)/np.log2(i+2) for i,x in enumerate(rank[:10]));ideal=sum((2**v-1)/np.log2(i+2) for i,v in enumerate(sorted(rr.values(),reverse=True)[:10]))
       rows.append({'qid':qid,'ndcg10':dcg/ideal,'recall10':sum(hits[:10])/len(rr),'recall100':sum(hits)/len(rr),'relevant':len(rr),'hits10':sum(hits[:10])});runs[qid]=rank
     csvsave(pd.DataFrame(rows),out);save(R/(k+'_'+ds+'_top100.json'),runs)
     print('EVAL',k,ds,len(rows),'queries',len(docids),'docs',time.perf_counter()-t,'seconds',flush=True)
   del m;gc.collect()
   if not cpu:torch.cuda.empty_cache()
  except Exception:
   err=traceback.format_exc();print(err,flush=True);save(R/(k+('_cpu_error.json' if cpu else '_gpu_error.json')),{'error':err})
if __name__=='__main__':main()
