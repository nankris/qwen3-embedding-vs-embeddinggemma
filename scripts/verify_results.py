"""Check saved metric rows independently against qrels and ranked IDs."""
import json,csv,math
from pathlib import Path
R=Path('results');checked=[]
for p in sorted(R.glob('*_top100.json')):
 k,ds=p.stem.split('_',1);ds=ds.removesuffix('_top100')
 data=json.loads((R/(ds+'_data.json')).read_text());ranks=json.loads(p.read_text());rel={}
 for r in data['qrels']:
  if r['score']>0:rel.setdefault(r['qid'],{})[r['did']]=r['score']
 rows=list(csv.DictReader(open(R/(k+'_'+ds+'_metrics.csv'))));assert len(rows)==len(rel)==len(ranks)
 docs={x['id'] for x in data['corpus']}
 for row in rows:
  qid=row['qid'];rank=ranks[qid];rr=rel[qid];assert len(rank)==100==len(set(rank));assert set(rank)<=docs;assert qid not in rank
  hit10=len(set(rank[:10])&set(rr));hit100=len(set(rank)&set(rr));ideal=sum((2**v-1)/math.log2(i+2) for i,v in enumerate(sorted(rr.values(),reverse=True)[:10]));dcg=sum((2**rr.get(d,0)-1)/math.log2(i+2) for i,d in enumerate(rank[:10]))
  for metric,val in [('ndcg10',dcg/ideal),('recall10',hit10/len(rr)),('recall100',hit100/len(rr))]:assert abs(float(row[metric])-val)<1e-10,(p,qid,metric)
  assert 0<=float(row['recall10'])<=float(row['recall100'])<=1
 checked.append({'file':p.name,'queries_verified':len(rows),'corpus_docs':len(docs)})
(R/'verification.json').write_text(json.dumps(checked,indent=2));print(json.dumps(checked,indent=2))
