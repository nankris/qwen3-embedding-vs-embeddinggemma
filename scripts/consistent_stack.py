"""Repeat initial-stack Qwen/BGE on the same supported software stack as Google."""
import time,subprocess,shutil
from pathlib import Path
r=Path('/content/exp098');old=Path('/proc/4142/stat')
while old.exists():
 try:
  if old.read_text().split()[2]=='Z':break
 except FileNotFoundError:break
 time.sleep(3)
results=r/'results';archive=results/'initial_stack';archive.mkdir(exist_ok=True)
for k in ['qwen','bge']:
 for ds in ['scifact','nfcorpus','arguana']:
  for suffix in ['_metrics.csv','_top100.json','_docs.npy']:
   p=results/(k+'_'+ds+suffix)
   if p.exists() and not (archive/p.name).exists():p.rename(archive/p.name)
 p=results/(k+'_gpu_speed.json')
 if p.exists() and not (archive/p.name).exists():p.rename(archive/p.name)
 p=results/(k+'_model.json')
 if p.exists() and not (archive/p.name).exists():shutil.copy(p,archive/p.name)
with open(r/'run.log','a') as f:
 print('CONSISTENT STACK repeat: all final English results use Transformers main',file=f,flush=True)
 subprocess.run(['python','-u','/content/run_eval.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
(r/'complete.json').write_text('{"status":"all final GPU work complete"}')
print('ALL GPU WORK DONE',time.time(),flush=True)
