"""Audit the required common-stack repeats against the archived first pass."""
from pathlib import Path
import csv,json
R=Path('results');out=[]
for k in ['qwen','bge']:
 for ds in ['scifact','nfcorpus','arguana']:
  f=k+'_'+ds+'_metrics.csv';a={r['qid']:r for r in csv.DictReader(open(R/'initial_stack'/f))};b={r['qid']:r for r in csv.DictReader(open(R/f))};assert a.keys()==b.keys()
  metrics=['ndcg10','recall10','recall100'];out.append({'model':k,'dataset':ds,'queries':len(a),'changed_queries':sum(any(abs(float(a[i][m])-float(b[i][m]))>1e-12 for m in metrics) for i in a),'max_metric_abs_difference':max(abs(float(a[i][m])-float(b[i][m])) for i in a for m in metrics)})
(R/'software_stack_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
